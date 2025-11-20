# Generación de archivos de etiquetas para la vista derecha 

base_path = Path("..")
input_labels_dir = base_path / "processed_data" / "data_yolo" / "labels" / "left_labels"
output_labels_dir = base_path / "processed_data" / "data_yolo" / "labels" / "right_labels"
output_labels_dir.mkdir(parents=True, exist_ok=True)

print(f"Leyendo etiquetas de: {input_labels_dir.resolve()}")
print(f"Guardando nuevas etiquetas en: {output_labels_dir.resolve()}")


# Bucle de procesamiento y transformación 
label_files = sorted(list(input_labels_dir.glob("*.txt")))

for label_path in tqdm(label_files, desc="Generando etiquetas (derecha)", unit="file"):
    new_lines = []
   
    try:
        with open(label_path, "r") as f_in:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue 

                class_id_str = parts[0]
                x_c = float(parts[1])
                y_c_str = parts[2]
                w_str = parts[3]
                h_str = parts[4]

                # Aplicar la transformación: desplazar x_center en 0.5
                x_c_new = x_c + 0.5
                # Asegurarse de que el valor no exceda 1.0 (medida de seguridad)
                x_c_new = min(1.0, x_c_new)
                # Reconstruir la nueva línea con la coordenada x actualizada
                new_line = f"{class_id_str} {x_c_new:.6f} {y_c_str} {w_str} {h_str}\n"
                new_lines.append(new_line)
       
        output_path = output_labels_dir / label_path.name
        with open(output_path, "w") as f_out:
            f_out.writelines(new_lines)
           
    except Exception as e:
        print(f"Error procesando el archivo {label_path.name}: {e}")
