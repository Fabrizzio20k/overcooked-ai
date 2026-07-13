#!/usr/bin/env python
"""Terminal User Interface (TUI) for the Overcooked-AI Competition Runner."""

import os
import sys
import subprocess
import json
import time

# Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def draw_header():
    print(f"{CYAN}{BOLD}┌────────────────────────────────────────────────────────┐{RESET}")
    print(f"{CYAN}{BOLD}│      🍳  OVERCOOKED-AI COMPETITION RUNNER (TUI)  🍳      │{RESET}")
    print(f"{CYAN}{BOLD}└────────────────────────────────────────────────────────┘{RESET}")

def print_menu():
    print(f"\n{BOLD}Selecciona una opción:{RESET}\n")
    print(f" {GREEN}[1]{RESET} {BOLD}Escenario 1{RESET} - Asymmetric Advantages (Cebollas)")
    print(f"     {YELLOW}↳ Evalúa la planificación heurística de tu agente.{RESET}")
    print(f" {GREEN}[2]{RESET} {BOLD}Escenario 2{RESET} - Coordination Ring (Teammate Pegajoso)")
    print(f"     {YELLOW}↳ Evalúa la evasión dinámica de obstáculos.{RESET}")
    print(f" {GREEN}[3]{RESET} {BOLD}Escenario 3{RESET} - Counter Circuit (Recetas Mixtas Tomate+Cebolla)")
    print(f"     {YELLOW}↳ Evalúa la cooperación complementaria de ingredientes.{RESET}")
    print(f" {GREEN}[4]{RESET} {BOLD}Escenario 4{RESET} - Custom Layout (Scenario 4)")
    print(f"     {YELLOW}↳ Mapa personalizado con cebollas.{RESET}")
    print(f" {RED}[5]{RESET} {BOLD}Salir{RESET}")
    print()

def configure_yaml(file_path, use_ppo):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace use_ppo setting
    if use_ppo:
        content = content.replace("use_ppo: false", "use_ppo: true")
        content = content.replace("type: state\n  include_agent_index", "type: featurized\n  include_agent_index")
    else:
        content = content.replace("use_ppo: true", "use_ppo: false")
        content = content.replace("type: featurized\n  include_agent_index", "type: state\n  include_agent_index")
        
    with open(file_path, 'w') as f:
        f.write(content)

def run_command(command):
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    print(f"\n{CYAN}{BOLD}Ejecutando simulación...{RESET}")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    idx = 0
    while process.poll() is None:
        sys.stdout.write(f"\r{YELLOW}{spinner[idx % len(spinner)]} Ejecutando paso a paso...{RESET}")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
        
    sys.stdout.write("\r")
    sys.stdout.flush()
    
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def parse_evaluation_results(stdout):
    try:
        lines = stdout.split("\n")
        json_str = ""
        started = False
        for line in lines:
            if line.strip() == "{":
                started = True
            if started:
                json_str += line + "\n"
            if line.strip() == "}":
                break
        if json_str:
            return json.loads(json_str)
    except Exception:
        pass
    return None

def show_evaluation_summary(scenario, results, is_ppo):
    if not results:
        print(f"{RED}Error al analizar los resultados de la evaluación.{RESET}")
        return
        
    mean_score = results.get("mean_return_sparse", 0.0)
    std_score = results.get("std_return_sparse", 0.0)
    returns = results.get("returns_sparse", [])
    
    print(f"\n{GREEN}{BOLD}┌────────────────────────────────────────────────────────┐{RESET}")
    print(f"{GREEN}{BOLD}│               RESUMEN DE LA EVALUACIÓN                 │{RESET}")
    print(f"{GREEN}{BOLD}└────────────────────────────────────────────────────────┘{RESET}")
    print(f" • {BOLD}Cerebro Usado:{RESET} {YELLOW}{'Red Neuronal PPO' if is_ppo else 'Planificación Heurística A*'}{RESET}")
    print(f" • {BOLD}Puntaje Medio (Sparse Return):{RESET} {CYAN}{mean_score:.2f}{RESET}")
    print(f" • {BOLD}Desviación Estándar:{RESET} {CYAN}{std_score:.2f}{RESET}")
    print(f" • {BOLD}Intentos individuales:{RESET} {returns}")
    
    print(f"\n{MAGENTA}{BOLD}📋 ESTIMACIÓN DE CALIFICACIÓN (Según reglas del Profesor):{RESET}")
    
    if is_ppo and scenario in [1, 2, 3] and mean_score < 40.0:
        print(f" ↳ {RED}{BOLD}Nota: Muy Baja / Reprobado{RESET}")
        print(f"     {YELLOW}⚠ Razón: Tu modelo PPO de 1M de pasos no sabe cooperar con 'greedy_full_task'.{RESET}")
        return

    if scenario == 1:
        if mean_score == 0:
            print(f" ↳ {RED}{BOLD}Nota: 0{RESET} (No se entregaron sopas)")
        else:
            soups = mean_score / 20.0
            print(f" ↳ {GREEN}Sopas promedio estimadas:{RESET} {soups:.1f}")
            if mean_score >= 100:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 9 (Puestos 5-1 - ¡Top 5!){RESET}")
            else:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 6+ (Aprobado){RESET}")
                
    elif scenario == 2:
        if mean_score < 40.0:
            print(f" ↳ {RED}{BOLD}Reprobado{RESET} (Menos de 2 sopas promedio)")
        else:
            soups = mean_score / 20.0
            print(f" ↳ {GREEN}Sopas promedio estimadas:{RESET} {soups:.1f}")
            if mean_score >= 160:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 12 (Puestos 5-1 - ¡Top 5!){RESET}")
            elif mean_score >= 120:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 11 (Puestos 10-6){RESET}")
            elif mean_score >= 80:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 10 (Puestos 15-11){RESET}")
            else:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 9 (Aprobado simple){RESET}")
                
    elif scenario == 3:
        if mean_score < 40.0:
            print(f" ↳ {RED}{BOLD}No clasifica{RESET} (Menos de 2 sopas promedio)")
        else:
            print(f" ↳ {GREEN}Puntaje promedio obtenido:{RESET} {mean_score:.1f}")
            if mean_score >= 60.0:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 14 (Puestos 4-1 - ¡Top 4! ¡Clasifica!){RESET}")
            elif mean_score >= 50.0:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 13 (Puestos 8-5 - ¡Clasifica!){RESET}")
            else:
                print(f" ↳ {GREEN}{BOLD}Nota Estimada: 11-12 (Puestos 12-9 - ¡Clasifica!){RESET}")

    elif scenario == 4:
        if mean_score == 0:
            print(f" ↳ {RED}{BOLD}Sin puntaje{RESET} (No se entregaron sopas)")
        else:
            soups = mean_score / 20.0
            print(f" ↳ {GREEN}Sopas promedio estimadas:{RESET} {soups:.1f}")
            print(f" ↳ {GREEN}{BOLD}Puntaje: {mean_score:.1f}{RESET}")

def main():
    while True:
        clear_screen()
        draw_header()
        print_menu()
        
        choice = input(f"{BOLD}Ingresa tu opción (1-5): {RESET}").strip()
        
        if choice == '5':
            print(f"\n{GREEN}¡Gracias por usar Overcooked-AI Competition Runner! ¡Buena suerte en la entrega! 🍳{RESET}\n")
            break
            
        if choice not in ['1', '2', '3', '4']:
            print(f"\n{RED}Opción inválida. Presiona ENTER para continuar...{RESET}")
            input()
            continue
            
        scenario = int(choice)
        
        is_ppo = (scenario != 4)
        
        # Choose mode
        print(f"\n{BOLD}Selecciona el modo de ejecución:{RESET}")
        print(f"  {CYAN}[E]{RESET} {BOLD}Evaluación Headless{RESET} (Corre 10 rollouts y calcula estadísticas y notas)")
        print(f"  {CYAN}[P]{RESET} {BOLD}Visualización en Vivo{RESET} (Abre la ventana gráfica de Pygame para ver la partida)")
        
        mode = input(f"\n{BOLD}Ingresa modo (E/P): {RESET}").strip().upper()
        
        # Configure files
        play_path = f"configs/play_escenario{scenario}.yaml"
        eval_path = f"configs/evaluate_escenario{scenario}.yaml"
        
        configure_yaml(play_path, is_ppo)
        configure_yaml(eval_path, is_ppo)
        
        if mode == 'E':
            cmd = f"PYTHONPATH=. ./venv/bin/python -m src.evaluate --config {eval_path}"
            ret_code, stdout, stderr = run_command(cmd)
            if ret_code != 0:
                print(f"\n{RED}Error durante la ejecución:{RESET}")
                print(stderr)
            else:
                results = parse_evaluation_results(stdout)
                show_evaluation_summary(scenario, results, is_ppo)
                
        elif mode == 'P':
            print(f"\n{GREEN}Iniciando interfaz gráfica de Pygame...{RESET}")
            print(f"{YELLOW}Nota: Puedes cerrar la ventana de Pygame en cualquier momento para volver aquí.{RESET}")
            cmd = f"PYTHONPATH=. ./venv/bin/python -m src.run_game --config {play_path}"
            ret_code, stdout, stderr = run_command(cmd)
            if ret_code != 0:
                print(f"\n{RED}Error durante la ejecución:{RESET}")
                print(stderr)
            else:
                print(f"\n{GREEN}Sesión gráfica finalizada correctamente.{RESET}")
        else:
            print(f"\n{RED}Modo inválido.{RESET}")
            
        print(f"\nPresiona ENTER para volver al menú principal...")
        input()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{RED}Programa cancelado por el usuario.{RESET}\n")
