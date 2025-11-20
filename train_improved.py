#!/usr/bin/env python3
"""   
Script de entrenamiento para OOBNet Improved
(c) Research Group CAMMA, University of Strasbourg, IHU Strasbourg, France

Entrena el modelo mejorado con BiLSTM + Attention y guarda checkpoints.
"""

import argparse
import os
import sys
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
import matplotlib.pyplot as plt

# Importar modelos
from model_improved import build_model_improved, preprocess_augmented

# Configuración de GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU detectada: {gpus[0].name}")
        print(f"✓ Número de GPUs disponibles: {len(gpus)}")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
else:
    print("⚠ No se detectó GPU. Ejecutando en CPU.")


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generador de datos para entrenamiento en batches.
    Carga videos y extrae frames con sus etiquetas.
    """
    def __init__(self, video_paths, annotations, batch_size=32, 
                 input_shape=[64, 64], shuffle=True, augment=True):
        """
        Args:
            video_paths: Lista de rutas a videos
            annotations: Diccionario {video_path: lista_etiquetas_por_frame}
            batch_size: Tamaño del batch
            input_shape: Dimensiones de entrada
            shuffle: Si barajar datos cada época
            augment: Si aplicar data augmentation
        """
        self.video_paths = video_paths
        self.annotations = annotations
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = None
        self.on_epoch_end()
    
    def __len__(self):
        """Número de batches por época"""
        # Calcular total de frames
        total_frames = sum([len(self.annotations[vp]) for vp in self.video_paths])
        return int(np.ceil(total_frames / self.batch_size))
    
    def __getitem__(self, index):
        """Genera un batch de datos"""
        # Aquí deberías implementar la lógica de cargar frames
        # Por ahora, estructura básica
        X = np.zeros((self.batch_size, *self.input_shape, 3))
        y = np.zeros((self.batch_size, 1))
        
        # TODO: Cargar frames reales del video
        # Esta es una implementación simplificada
        
        return X, y
    
    def on_epoch_end(self):
        """Actualiza índices después de cada época"""
        self.indexes = np.arange(len(self.video_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def create_training_callbacks(checkpoint_dir, model_name='oobnet_improved'):
    """
    Crea callbacks para el entrenamiento.
    
    Returns:
        Lista de callbacks de Keras
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'logs'), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Guardar mejor modelo
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        ),
        
        # Guardar checkpoints regulares
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'{model_name}_epoch_{{epoch:02d}}_val_loss_{{val_loss:.4f}}.h5'),
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            save_freq='epoch',
            verbose=0
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reducir learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=os.path.join(checkpoint_dir, 'logs', timestamp),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV Logger
        CSVLogger(
            filename=os.path.join(checkpoint_dir, f'{model_name}_training_log_{timestamp}.csv'),
            separator=',',
            append=False
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path):
    """
    Genera gráficas del historial de entrenamiento.
    
    Args:
        history: Historial de entrenamiento de Keras
        save_path: Ruta donde guardar las gráficas
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    if 'accuracy' in history.history:
        axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráficas guardadas en: {save_path}")


def train_model(train_data_path, val_data_path, checkpoint_dir='./checkpoints',
                batch_size=32, epochs=100, initial_lr=1e-3,
                input_shape=[64, 64], dropout_rate=0.2, lstm_units=320):
    """
    Función principal de entrenamiento.
    
    Args:
        train_data_path: Ruta a datos de entrenamiento
        val_data_path: Ruta a datos de validación
        checkpoint_dir: Directorio para guardar checkpoints
        batch_size: Tamaño del batch
        epochs: Número de épocas
        initial_lr: Learning rate inicial
        input_shape: Dimensiones de entrada
        dropout_rate: Tasa de dropout
        lstm_units: Unidades LSTM por dirección
    """
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO DE OOBNet IMPROVED")
    print("="*70)
    
    # Crear directorios
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Construir modelo
    print("\n📦 Construyendo modelo...")
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model = build_model_improved(
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units
        )
    
    # Compilar modelo
    print("⚙️  Compilando modelo...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    model.summary()
    
    # Guardar configuración
    config = {
        'input_shape': input_shape,
        'dropout_rate': dropout_rate,
        'lstm_units': lstm_units,
        'batch_size': batch_size,
        'initial_lr': initial_lr,
        'epochs': epochs,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n✓ Configuración guardada en: {checkpoint_dir}/training_config.json")
    
    # TODO: Cargar datos reales
    # Por ahora, creamos datos dummy para demostración
    print("\n⚠️  NOTA: Usando datos dummy para demostración")
    print("   Reemplaza esto con tu generador de datos real.\n")
    
    # Datos dummy (REEMPLAZAR CON TUS DATOS REALES)
    X_train = np.random.rand(1000, *input_shape, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)
    X_val = np.random.rand(200, *input_shape, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, (200, 1)).astype(np.float32)
    
    # Crear callbacks
    callbacks = create_training_callbacks(checkpoint_dir, model_name='oobnet_improved')
    
    # Entrenar
    print("\n🚀 Iniciando entrenamiento...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar historial
    history_path = os.path.join(checkpoint_dir, 'training_history.npy')
    np.save(history_path, history.history)
    print(f"\n✓ Historial guardado en: {history_path}")
    
    # Generar gráficas
    plot_path = os.path.join(checkpoint_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Guardar modelo final
    final_model_path = os.path.join(checkpoint_dir, 'oobnet_improved_final.h5')
    model.save_weights(final_model_path)
    print(f"✓ Modelo final guardado en: {final_model_path}")
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar OOBNet Improved')
    
    parser.add_argument('--train_data', type=str, required=True,
                        help='Ruta a datos de entrenamiento')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Ruta a datos de validación')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_improved',
                        help='Directorio para guardar checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate inicial')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Tasa de dropout')
    parser.add_argument('--lstm_units', type=int, default=320,
                        help='Unidades LSTM por dirección')
    parser.add_argument('--input_size', type=int, default=64,
                        help='Tamaño de entrada (cuadrado)')
    
    args = parser.parse_args()
    
    # Entrenar modelo
    train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        initial_lr=args.lr,
        input_shape=[args.input_size, args.input_size],
        dropout_rate=args.dropout,
        lstm_units=args.lstm_units
    )
