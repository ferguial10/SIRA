#!/usr/bin/env python3
"""   
Script de evaluación y comparación de modelos OOBNet
(c) Research Group CAMMA, University of Strasbourg, IHU Strasbourg, France

Compara el rendimiento del modelo original vs. modelo mejorado en un conjunto de test.
Genera métricas detalladas y visualizaciones para el paper.
"""

import argparse
import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
import cv2
import tensorflow as tf

# Importar modelos
from model import build_model as build_model_original
from model_improved import build_model_improved, preprocess

# Configuración
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sns.set_style('whitegrid')

# GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU detectada: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")


def load_ground_truth(csv_path):
    """
    Carga el ground truth desde un archivo CSV.
    
    Args:
        csv_path: Ruta al CSV con columnas FRAME_ID y OOB (0 o 1)
        
    Returns:
        numpy array con etiquetas binarias
    """
    df = pd.read_csv(csv_path)
    return df['OOB'].values


def predict_video(model, video_path, model_name="model"):
    """
    Ejecuta predicción frame por frame en un video.
    
    Args:
        model: Modelo de Keras cargado
        video_path: Ruta al video
        model_name: Nombre del modelo (para logging)
        
    Returns:
        numpy array con predicciones (probabilidades)
    """
    video_in = cv2.VideoCapture(video_path)
    if not video_in.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    video_nframes = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = []
    
    print(f"\n🔮 Prediciendo con {model_name}: {video_nframes} frames...")
    
    ok, frame = video_in.read()
    i = 0
    
    while ok:
        preprocessed = preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            pred = model(preprocessed, training=False)
        
        predictions.append(pred.numpy()[0, 0, 0] if len(pred.shape) > 2 else pred.numpy()[0, 0])
        
        if i > 0 and i % 500 == 0:
            print(f"   Procesado: {i}/{video_nframes} frames ({i/video_nframes*100:.1f}%)")
        
        ok, frame = video_in.read()
        i += 1
    
    video_in.release()
    print(f"   ✓ Completado: {len(predictions)} predicciones")
    
    return np.array(predictions)


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_true: Ground truth (0 o 1)
        y_pred_proba: Predicciones de probabilidad
        threshold: Umbral para binarización
        
    Returns:
        Diccionario con todas las métricas
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Métricas básicas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Especificidad
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC y AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }
    
    return metrics


def plot_comparison(metrics_original, metrics_improved, save_dir):
    """
    Genera gráficas comparativas entre modelo original y mejorado.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Comparación de métricas principales
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC AUC']
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    original_values = [metrics_original[m] for m in metrics_names]
    improved_values = [metrics_improved[m] for m in metrics_names]
    
    bars1 = ax.bar(x - width/2, original_values, width, label='Original', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, improved_values, width, label='Improved (BiLSTM+Attention)', alpha=0.8, color='#e74c3c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparación de Métricas: Original vs. Improved', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Añadir valores sobre las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: metrics_comparison.png")
    plt.close()
    
    # 2. Matrices de confusión lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (metrics, title) in enumerate([
        (metrics_original, 'Original Model'),
        (metrics_improved, 'Improved Model (BiLSTM+Attention)')
    ]):
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   cbar_kws={'label': 'Frames'})
        axes[idx].set_title(title, fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
        axes[idx].set_xticklabels(['In-Body', 'Out-of-Body'])
        axes[idx].set_yticklabels(['In-Body', 'Out-of-Body'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: confusion_matrices.png")
    plt.close()
    
    # 3. Curvas ROC
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(metrics_original['fpr'], metrics_original['tpr'], 
           label=f"Original (AUC = {metrics_original['roc_auc']:.3f})",
           color='#3498db', linewidth=2.5)
    ax.plot(metrics_improved['fpr'], metrics_improved['tpr'], 
           label=f"Improved (AUC = {metrics_improved['roc_auc']:.3f})",
           color='#e74c3c', linewidth=2.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: roc_curves.png")
    plt.close()
    
    # 4. Precision-Recall curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(metrics_original['recall_curve'], metrics_original['precision_curve'],
           label=f"Original (AP = {metrics_original['avg_precision']:.3f})",
           color='#3498db', linewidth=2.5)
    ax.plot(metrics_improved['recall_curve'], metrics_improved['precision_curve'],
           label=f"Improved (AP = {metrics_improved['avg_precision']:.3f})",
           color='#e74c3c', linewidth=2.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: precision_recall_curves.png")
    plt.close()


def generate_report(metrics_original, metrics_improved, save_path):
    """
    Genera reporte detallado en formato texto y JSON.
    """
    # Calcular mejoras porcentuales
    improvements = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']:
        orig_val = metrics_original[metric]
        imp_val = metrics_improved[metric]
        improvement = ((imp_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
        improvements[metric] = {
            'original': float(orig_val),
            'improved': float(imp_val),
            'improvement_percent': float(improvement)
        }
    
    # Reporte en texto
    report_txt = f"""
{'='*80}
REPORTE DE COMPARACIÓN: OOBNet Original vs. OOBNet Improved
{'='*80}

Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ARQUITECTURA DEL MODELO MEJORADO:
- Bidirectional LSTM (2 × 320 units)
- Attention Mechanism
- Dropout regularization
- Layer Normalization

{'='*80}
MÉTRICAS COMPARATIVAS
{'='*80}

Métrica          | Original | Improved | Mejora (%)
-----------------+----------+----------+------------
"""
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']:
        orig = improvements[metric]['original']
        imp = improvements[metric]['improved']
        pct = improvements[metric]['improvement_percent']
        sign = '+' if pct >= 0 else ''
        report_txt += f"{metric.capitalize():16s} | {orig:8.4f} | {imp:8.4f} | {sign}{pct:+7.2f}%\n"
    
    report_txt += f"""
{'='*80}
MATRIZ DE CONFUSIÓN - MODELO ORIGINAL
{'='*80}

                 Predicted
                 In-Body    Out-of-Body
Actual  In-Body     {metrics_original['true_negatives']:6d}        {metrics_original['false_positives']:6d}
        Out-of-Body {metrics_original['false_negatives']:6d}        {metrics_original['true_positives']:6d}

{'='*80}
MATRIZ DE CONFUSIÓN - MODELO MEJORADO
{'='*80}

                 Predicted
                 In-Body    Out-of-Body
Actual  In-Body     {metrics_improved['true_negatives']:6d}        {metrics_improved['false_positives']:6d}
        Out-of-Body {metrics_improved['false_negatives']:6d}        {metrics_improved['true_positives']:6d}

{'='*80}
ANÁLISIS DE ERRORES
{'='*80}

MODELO ORIGINAL:
  - Falsos Positivos: {metrics_original['false_positives']} frames
  - Falsos Negativos: {metrics_original['false_negatives']} frames
  - Error Total: {metrics_original['false_positives'] + metrics_original['false_negatives']} frames

MODELO MEJORADO:
  - Falsos Positivos: {metrics_improved['false_positives']} frames
  - Falsos Negativos: {metrics_improved['false_negatives']} frames
  - Error Total: {metrics_improved['false_positives'] + metrics_improved['false_negatives']} frames

Reducción de errores: {((metrics_original['false_positives'] + metrics_original['false_negatives']) - (metrics_improved['false_positives'] + metrics_improved['false_negatives']))} frames

{'='*80}
CONCLUSIÓN
{'='*80}

"""
    
    # Determinar si hubo mejora significativa
    avg_improvement = np.mean([improvements[m]['improvement_percent'] 
                               for m in ['accuracy', 'precision', 'recall', 'f1_score']])
    
    if avg_improvement > 5:
        report_txt += "✅ El modelo mejorado muestra MEJORA SIGNIFICATIVA sobre el original.\n"
    elif avg_improvement > 1:
        report_txt += "✓ El modelo mejorado muestra mejora moderada sobre el original.\n"
    elif avg_improvement > -1:
        report_txt += "≈ El modelo mejorado tiene rendimiento similar al original.\n"
    else:
        report_txt += "⚠️  El modelo mejorado muestra degradación respecto al original.\n"
    
    report_txt += f"\nMejora promedio en métricas principales: {avg_improvement:+.2f}%\n"
    report_txt += "\n" + "="*80 + "\n"
    
    # Guardar reporte texto
    txt_path = save_path.replace('.json', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_txt)
    print(f"✓ Reporte guardado: {txt_path}")
    
    # Guardar JSON
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'improvements': improvements,
        'average_improvement_percent': float(avg_improvement),
        'original_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                           for k, v in metrics_original.items() 
                           if k not in ['confusion_matrix', 'fpr', 'tpr', 'precision_curve', 'recall_curve']},
        'improved_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                           for k, v in metrics_improved.items()
                           if k not in ['confusion_matrix', 'fpr', 'tpr', 'precision_curve', 'recall_curve']}
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
    print(f"✓ JSON guardado: {save_path}")
    
    return report_txt


def evaluate_models(ckpt_original, ckpt_improved, video_path, ground_truth_csv, output_dir='./evaluation_results'):
    """
    Función principal de evaluación.
    """
    print("\n" + "="*80)
    print("EVALUACIÓN COMPARATIVA DE MODELOS OOBNet")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar ground truth
    print("\n📂 Cargando ground truth...")
    y_true = load_ground_truth(ground_truth_csv)
    print(f"   ✓ Cargadas {len(y_true)} etiquetas")
    print(f"   - Frames In-Body: {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.1f}%)")
    print(f"   - Frames Out-of-Body: {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)")
    
    # Cargar y evaluar modelo original
    print("\n" + "="*80)
    print("EVALUANDO MODELO ORIGINAL")
    print("="*80)
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model_original = build_model_original()
        model_original.load_weights(ckpt_original)
    
    y_pred_original = predict_video(model_original, video_path, "Original")
    metrics_original = calculate_metrics(y_true, y_pred_original)
    
    # Cargar y evaluar modelo mejorado
    print("\n" + "="*80)
    print("EVALUANDO MODELO MEJORADO (BiLSTM + Attention)")
    print("="*80)
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model_improved = build_model_improved()
        model_improved.load_weights(ckpt_improved)
    
    y_pred_improved = predict_video(model_improved, video_path, "Improved")
    metrics_improved = calculate_metrics(y_true, y_pred_improved)
    
    # Guardar predicciones
    pred_df = pd.DataFrame({
        'FRAME_ID': range(len(y_true)),
        'GROUND_TRUTH': y_true,
        'PRED_ORIGINAL': y_pred_original,
        'PRED_IMPROVED': y_pred_improved,
        'PRED_ORIGINAL_BINARY': (y_pred_original >= 0.5).astype(int),
        'PRED_IMPROVED_BINARY': (y_pred_improved >= 0.5).astype(int)
    })
    pred_df.to_csv(os.path.join(output_dir, 'predictions_comparison.csv'), index=False)
    print(f"\n✓ Predicciones guardadas: {output_dir}/predictions_comparison.csv")
    
    # Generar visualizaciones
    print("\n📊 Generando visualizaciones...")
    plot_comparison(metrics_original, metrics_improved, output_dir)
    
    # Generar reporte
    print("\n📝 Generando reporte...")
    report = generate_report(metrics_original, metrics_improved, 
                            os.path.join(output_dir, 'evaluation_report.json'))
    
    print("\n" + report)
    
    print("\n" + "="*80)
    print("✅ EVALUACIÓN COMPLETADA")
    print("="*80)
    print(f"\nResultados guardados en: {output_dir}/")
    print("Archivos generados:")
    print("  - evaluation_report.txt")
    print("  - evaluation_report.json")
    print("  - predictions_comparison.csv")
    print("  - metrics_comparison.png")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")
    print("  - precision_recall_curves.png")
    print()


if __name__ == '__main__':
    def file_path(string):
        if os.path.isfile(string):
            return string
        else:
            raise FileNotFoundError(f"Archivo no encontrado: {string}")
    
    parser = argparse.ArgumentParser(
        description='Evaluar y comparar OOBNet Original vs. Improved',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--ckpt_original', type=file_path, required=True,
                       help='Ruta a los pesos del modelo original (.h5)')
    parser.add_argument('--ckpt_improved', type=file_path, required=True,
                       help='Ruta a los pesos del modelo mejorado (.h5)')
    parser.add_argument('--video', type=file_path, required=True,
                       help='Ruta al video de test')
    parser.add_argument('--ground_truth', type=file_path, required=True,
                       help='Ruta al CSV con ground truth (columnas: FRAME_ID, OOB)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    evaluate_models(
        ckpt_original=args.ckpt_original,
        ckpt_improved=args.ckpt_improved,
        video_path=args.video,
        ground_truth_csv=args.ground_truth,
        output_dir=args.output_dir
    )
