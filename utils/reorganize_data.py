import os
import shutil
import glob


def reorganize_data(source_dir, dest_dir):
    """
    Extrae solo los archivos .npz esenciales de todas las carpetas
    y los organiza por el nombre del escenario (layout).
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    npz_files = glob.glob(os.path.join(source_dir, "**", "*.npz"), recursive=True)
    print(f"Se encontraron {len(npz_files)} archivos .npz esenciales.")

    for filepath in npz_files:
        # El nombre del archivo suele ser 'escenario_fecha_hora.npz'
        # Ejemplo: asymmetric_advantages_20260522_113242.npz
        filename = os.path.basename(filepath)
        layout_name = filename.rsplit("_", 2)[0]  # Extraer el nombre del layout

        layout_dir = os.path.join(dest_dir, layout_name)
        if not os.path.exists(layout_dir):
            os.makedirs(layout_dir)

        dest_path = os.path.join(layout_dir, filename)
        shutil.copy(filepath, dest_path)

    print(f"Archivos reorganizados exitosamente en '{dest_dir}'.")


if __name__ == "__main__":
    reorganize_data("data", "data_clean")
