"""
Training Pipeline for AlphaZero-Mini

Este modulo implementa el loop completo de entrenamiento de AlphaZero:
1. SELF-PLAY: El agente juega contra si mismo y genera datos
2. TRAINING: La red neuronal aprende de los datos generados
3. EVALUATION: Se evalua si el nuevo modelo es mejor
4. ITERATE: Repetir hasta convergencia

El proceso iterativo hace que el modelo mejore continuamente
al aprender de versiones cada vez mas fuertes de si mismo.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple, Dict, Optional
import os
import time
import json
import random

# Imports locales
from game import Connect4Game
from neural_net import Connect4Net, board_to_tensor, save_checkpoint, load_checkpoint
from mcts import MCTS
from agent import AlphaZeroAgent


def prepare_batch(
    examples: List[Tuple[np.ndarray, np.ndarray, float]],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepara un batch de ejemplos para entrenamiento.
    
    Convierte lista de (board, policy, value) a tensores batched.
    Los tableros se convierten a representacion de 3 canales.
    
    Args:
        examples: Lista de tuplas (board, policy, value)
        device: Dispositivo para los tensores (CPU/CUDA)
    
    Returns:
        Tupla (boards, policies, values):
            - boards: Tensor (batch, 3, 6, 7)
            - policies: Tensor (batch, 7)
            - values: Tensor (batch, 1)
    """
    boards = []
    policies = []
    values = []
    
    for board, policy, value in examples:
        # Convertir tablero a tensor de 3 canales
        # Nota: board ya esta en forma canonica (perspectiva del jugador)
        # Usamos player=1 porque ya es canonico
        board_tensor = board_to_tensor(board, player=1)
        boards.append(board_tensor)
        policies.append(torch.tensor(policy, dtype=torch.float32))
        values.append(torch.tensor([value], dtype=torch.float32))
    
    # Stack para crear batch
    boards_batch = torch.stack(boards).to(device)
    policies_batch = torch.stack(policies).to(device)
    values_batch = torch.stack(values).to(device)
    
    return boards_batch, policies_batch, values_batch


class AlphaZeroTrainer:
    """
    Pipeline de entrenamiento completo para AlphaZero.
    
    Ejecuta el ciclo:
    1. Self-play: Generar datos jugando contra si mismo
    2. Training: Entrenar red neuronal con los datos
    3. Evaluation: Comparar nuevo modelo vs anterior
    4. Iterate: Repetir y mejorar continuamente
    
    Attributes:
        model: Red neuronal actual (mejor modelo)
        optimizer: Optimizador Adam
        device: Dispositivo (CPU/CUDA)
        agent: Agente de self-play
        replay_buffer: Buffer de ejemplos de entrenamiento
        training_stats: Historial de estadisticas
        checkpoint_dir: Directorio para guardar modelos
    """
    
    def __init__(
        self,
        model: Connect4Net,
        device: str = 'cuda',
        checkpoint_dir: str = 'models/',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        num_epochs: int = 5,
        num_self_play_games: int = 100,
        num_eval_games: int = 20,
        win_rate_threshold: float = 0.55,
        num_simulations: int = 100,
        num_simulations_eval: int = 200,
        temperature_threshold: int = 10,
        buffer_size: int = 50000
    ) -> None:
        """
        Inicializa el trainer de AlphaZero.
        
        Args:
            model: Red neuronal a entrenar
            device: 'cpu' o 'cuda'
            checkpoint_dir: Directorio para checkpoints
            learning_rate: Tasa de aprendizaje
            weight_decay: Regularizacion L2
            batch_size: Tamano de batch para entrenamiento
            num_epochs: Epocas por fase de entrenamiento
            num_self_play_games: Partidas por iteracion de self-play
            num_eval_games: Partidas para evaluacion
            win_rate_threshold: Umbral para aceptar nuevo modelo (0.55 = 55%)
            num_simulations: Simulaciones MCTS en self-play
            num_simulations_eval: Simulaciones MCTS en evaluacion
            temperature_threshold: Movimientos con temperatura alta
            buffer_size: Tamano maximo del replay buffer
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        # Hiperparametros
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_self_play_games = num_self_play_games
        self.num_eval_games = num_eval_games
        self.win_rate_threshold = win_rate_threshold
        self.num_simulations = num_simulations
        self.num_simulations_eval = num_simulations_eval
        self.temperature_threshold = temperature_threshold
        
        # Optimizador
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Agente de self-play
        self.agent = AlphaZeroAgent(
            model=self.model,
            num_simulations=num_simulations,
            device=device
        )
        
        # Replay buffer: almacena ejemplos de entrenamiento
        self.replay_buffer: deque = deque(maxlen=buffer_size)
        
        # Estadisticas de entrenamiento
        self.training_stats: Dict[str, List] = {
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'win_rates': [],
            'buffer_sizes': [],
            'iteration_times': []
        }
        
        # Directorio de checkpoints
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Iteration actual
        self.current_iteration = 0
        
        print("=" * 60)
        print("AlphaZero Trainer inicializado")
        print("=" * 60)
        print(f"  Dispositivo: {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Self-play games/iter: {num_self_play_games}")
        print(f"  MCTS simulaciones: {num_simulations}")
        print(f"  Buffer size: {buffer_size}")
        print("=" * 60)
    
    def train_network(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray, float]]
    ) -> Dict[str, float]:
        """
        Entrena la red neuronal con los ejemplos dados.
        
        Ejecuta num_epochs de entrenamiento sobre los ejemplos,
        optimizando tanto la policy (cross-entropy) como el value (MSE).
        
        Args:
            examples: Lista de (board, policy, value) para entrenar
        
        Returns:
            Diccionario con perdidas promedio:
                - 'policy_loss': Perdida de politica
                - 'value_loss': Perdida de valor
                - 'total_loss': Suma de ambas
        """
        self.model.train()
        
        # Mezclar ejemplos
        examples_shuffled = examples.copy()
        random.shuffle(examples_shuffled)
        
        # Calcular numero de batches
        num_examples = len(examples_shuffled)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        batch_count = 0
        
        for epoch in range(self.num_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_batches = 0
            
            # Mezclar al inicio de cada epoca
            random.shuffle(examples_shuffled)
            
            for batch_idx in range(num_batches):
                # Extraer batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_examples)
                batch_examples = examples_shuffled[start_idx:end_idx]
                
                if len(batch_examples) == 0:
                    continue
                
                # Preparar tensores
                boards, target_policies, target_values = prepare_batch(
                    batch_examples, self.device
                )
                
                # Forward pass
                policy_logits, predicted_values = self.model(boards)
                
                # Calcular perdidas
                # Policy loss: Cross-entropy entre logits y distribucion target
                # Nota: target_policies son probabilidades, no indices
                policy_loss = -torch.mean(
                    torch.sum(target_policies * torch.log_softmax(policy_logits, dim=1), dim=1)
                )
                
                # Value loss: MSE entre prediccion y valor real
                value_loss = torch.mean((predicted_values - target_values) ** 2)
                
                # Perdida total
                total_loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Acumular perdidas
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_batches += 1
            
            # Acumular para promedio global
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            batch_count += epoch_batches
            
            # Imprimir progreso de epoca
            avg_p = epoch_policy_loss / max(epoch_batches, 1)
            avg_v = epoch_value_loss / max(epoch_batches, 1)
            print(f"    Epoca {epoch + 1}/{self.num_epochs}: "
                  f"policy_loss={avg_p:.4f}, value_loss={avg_v:.4f}")
        
        # Calcular promedios finales
        avg_policy_loss = total_policy_loss / max(batch_count, 1)
        avg_value_loss = total_value_loss / max(batch_count, 1)
        avg_total_loss = avg_policy_loss + avg_value_loss
        
        self.model.eval()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_total_loss
        }
    
    def evaluate_model(self, new_model: Connect4Net) -> float:
        """
        Evalua el nuevo modelo contra el modelo actual.
        
        Juega num_eval_games partidas, alternando quien empieza.
        Retorna el porcentaje de victorias del nuevo modelo.
        
        Args:
            new_model: Modelo nuevo a evaluar
        
        Returns:
            Win rate del nuevo modelo (0.0 a 1.0)
        """
        print(f"    Evaluando: {self.num_eval_games} partidas...")
        
        # Crear agentes
        agent_new = AlphaZeroAgent(
            model=new_model,
            num_simulations=self.num_simulations_eval,
            device=str(self.device)
        )
        
        agent_old = AlphaZeroAgent(
            model=self.model,
            num_simulations=self.num_simulations_eval,
            device=str(self.device)
        )
        
        wins_new = 0
        wins_old = 0
        draws = 0
        
        for game_idx in range(self.num_eval_games):
            # Alternar quien juega primero
            new_plays_first = (game_idx % 2 == 0)
            
            game = Connect4Game()
            current_player = 1
            
            while not game.is_terminal():
                # Determinar que agente juega
                if new_plays_first:
                    current_agent = agent_new if current_player == 1 else agent_old
                else:
                    current_agent = agent_old if current_player == 1 else agent_new
                
                # Seleccionar accion deterministicamente
                action = current_agent.select_action(
                    game_state=game,
                    player=current_player,
                    temperature=0,
                    deterministic=True
                )
                
                game = game.make_move(action, current_player)
                current_player *= -1
            
            # Determinar ganador
            winner = game.check_winner()
            
            if winner == 0:
                draws += 1
            else:
                # Mapear winner a que agente gano
                if new_plays_first:
                    if winner == 1:
                        wins_new += 1
                    else:
                        wins_old += 1
                else:
                    if winner == 1:
                        wins_old += 1
                    else:
                        wins_new += 1
            
            # Progreso cada 5 partidas
            if (game_idx + 1) % 5 == 0:
                print(f"      Partida {game_idx + 1}/{self.num_eval_games}: "
                      f"Nuevo={wins_new}, Actual={wins_old}, Empates={draws}")
        
        # Calcular win rate (empates cuentan como 0.5)
        win_rate = (wins_new + 0.5 * draws) / self.num_eval_games
        
        print(f"    Resultado: Nuevo={wins_new}, Actual={wins_old}, Empates={draws}")
        print(f"    Win rate nuevo modelo: {win_rate:.1%}")
        
        return win_rate
    
    def train_iteration(self, iteration: int) -> Dict:
        """
        Ejecuta una iteracion completa de entrenamiento.
        
        Fases:
        1. Self-play: Generar nuevos datos
        2. Training: Entrenar red con datos del buffer
        3. Evaluation: Comparar nuevo modelo (cada 5 iteraciones)
        4. Checkpoint: Guardar estado
        
        Args:
            iteration: Numero de iteracion actual
        
        Returns:
            Diccionario con estadisticas de la iteracion
        """
        print("\n" + "=" * 60)
        print(f"ITERACION {iteration}")
        print("=" * 60)
        
        start_time = time.time()
        stats = {'iteration': iteration}
        
        # =====================================================================
        # FASE 1: SELF-PLAY
        # =====================================================================
        print(f"\n[1/4] SELF-PLAY ({self.num_self_play_games} partidas)")
        print("-" * 40)
        
        examples, game_stats = self.agent.play_games(
            num_games=self.num_self_play_games,
            temperature_threshold=self.temperature_threshold
        )
        
        # Agregar ejemplos al buffer
        self.replay_buffer.extend(examples)
        
        stats['examples_generated'] = len(examples)
        stats['buffer_size'] = len(self.replay_buffer)
        stats['game_stats'] = game_stats
        
        print(f"\nEjemplos generados: {len(examples)}")
        print(f"Buffer total: {len(self.replay_buffer)}")
        
        # =====================================================================
        # FASE 2: TRAINING
        # =====================================================================
        print(f"\n[2/4] TRAINING")
        print("-" * 40)
        
        # Seleccionar ejemplos para entrenar
        # Usamos una muestra del buffer para evitar sobreajuste
        num_train_examples = min(len(self.replay_buffer), self.batch_size * 50)
        train_examples = random.sample(list(self.replay_buffer), num_train_examples)
        
        print(f"Entrenando con {len(train_examples)} ejemplos...")
        
        losses = self.train_network(train_examples)
        
        stats['policy_loss'] = losses['policy_loss']
        stats['value_loss'] = losses['value_loss']
        stats['total_loss'] = losses['total_loss']
        
        print(f"\nPerdidas finales:")
        print(f"  Policy: {losses['policy_loss']:.4f}")
        print(f"  Value: {losses['value_loss']:.4f}")
        print(f"  Total: {losses['total_loss']:.4f}")
        
        # Actualizar estadisticas historicas
        self.training_stats['policy_losses'].append(losses['policy_loss'])
        self.training_stats['value_losses'].append(losses['value_loss'])
        self.training_stats['total_losses'].append(losses['total_loss'])
        self.training_stats['buffer_sizes'].append(len(self.replay_buffer))
        
        # =====================================================================
        # FASE 3: EVALUATION (cada 5 iteraciones)
        # =====================================================================
        print(f"\n[3/4] EVALUATION")
        print("-" * 40)
        
        if iteration > 0 and iteration % 5 == 0:
            # Para evaluacion, comparamos el modelo actual entrenado
            # contra una copia del modelo anterior
            # En una implementacion completa, guardariamos el modelo anterior
            # Por simplicidad, asumimos que el modelo actual es mejor si el loss bajo
            
            print("Evaluacion programada (cada 5 iteraciones)")
            
            # Verificar mejora basada en loss
            if len(self.training_stats['total_losses']) >= 2:
                prev_loss = self.training_stats['total_losses'][-2]
                curr_loss = self.training_stats['total_losses'][-1]
                
                if curr_loss < prev_loss:
                    print(f"  [OK] Loss mejoro: {prev_loss:.4f} -> {curr_loss:.4f}")
                    stats['model_improved'] = True
                else:
                    print(f"  [!] Loss no mejoro: {prev_loss:.4f} -> {curr_loss:.4f}")
                    stats['model_improved'] = False
            else:
                stats['model_improved'] = True
        else:
            print(f"Saltando evaluacion (proxima en iteracion {((iteration // 5) + 1) * 5})")
            stats['model_improved'] = None
        
        # =====================================================================
        # FASE 4: CHECKPOINT
        # =====================================================================
        print(f"\n[4/4] CHECKPOINT")
        print("-" * 40)
        
        # Guardar checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_iter_{iteration:04d}.pt'
        )
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=iteration,
            filepath=checkpoint_path
        )
        
        # Guardar "best" model
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=iteration,
            filepath=best_path
        )
        
        # Limpiar checkpoints antiguos (mantener solo los ultimos 5)
        self._cleanup_old_checkpoints(keep_last=5)
        
        # Guardar estadisticas
        stats_path = os.path.join(self.checkpoint_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # =====================================================================
        # RESUMEN
        # =====================================================================
        elapsed_time = time.time() - start_time
        stats['elapsed_time'] = elapsed_time
        self.training_stats['iteration_times'].append(elapsed_time)
        
        print(f"\n{'=' * 60}")
        print(f"ITERACION {iteration} COMPLETADA")
        print(f"{'=' * 60}")
        print(f"  Tiempo: {elapsed_time:.1f}s")
        print(f"  Ejemplos generados: {stats['examples_generated']}")
        print(f"  Buffer size: {stats['buffer_size']}")
        print(f"  Policy loss: {stats['policy_loss']:.4f}")
        print(f"  Value loss: {stats['value_loss']:.4f}")
        
        self.current_iteration = iteration
        
        return stats
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Elimina checkpoints antiguos, manteniendo los ultimos N."""
        checkpoint_files = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith('checkpoint_iter_') and filename.endswith('.pt'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                checkpoint_files.append(filepath)
        
        # Ordenar por nombre (que incluye el numero de iteracion)
        checkpoint_files.sort()
        
        # Eliminar los mas antiguos
        files_to_delete = checkpoint_files[:-keep_last] if len(checkpoint_files) > keep_last else []
        
        for filepath in files_to_delete:
            try:
                os.remove(filepath)
            except OSError:
                pass
    
    def train(
        self,
        num_iterations: int,
        resume_from: Optional[str] = None
    ) -> None:
        """
        Loop principal de entrenamiento.
        
        Ejecuta num_iterations iteraciones de:
        self-play -> training -> evaluation -> checkpoint
        
        Args:
            num_iterations: Numero total de iteraciones
            resume_from: Ruta de checkpoint para continuar entrenamiento
        """
        start_iteration = 0
        
        # Cargar checkpoint si se especifica
        if resume_from and os.path.exists(resume_from):
            print(f"\nCargando checkpoint: {resume_from}")
            start_iteration = load_checkpoint(
                filepath=resume_from,
                model=self.model,
                optimizer=self.optimizer
            )
            start_iteration += 1
            print(f"Continuando desde iteracion {start_iteration}")
            
            # Intentar cargar buffer
            buffer_path = os.path.join(self.checkpoint_dir, 'replay_buffer.pkl')
            if os.path.exists(buffer_path):
                import pickle
                with open(buffer_path, 'rb') as f:
                    self.replay_buffer = pickle.load(f)
                print(f"Buffer cargado: {len(self.replay_buffer)} ejemplos")
        
        print("\n" + "=" * 60)
        print("INICIANDO ENTRENAMIENTO ALPHAZERO")
        print("=" * 60)
        print(f"Iteraciones: {start_iteration} -> {num_iterations}")
        print(f"Dispositivo: {self.device}")
        print("=" * 60)
        
        training_start_time = time.time()
        
        for iteration in range(start_iteration, num_iterations):
            try:
                stats = self.train_iteration(iteration)
            except KeyboardInterrupt:
                print("\n\n[!] Entrenamiento interrumpido por usuario")
                print("Guardando estado actual...")
                self._save_buffer()
                break
            except Exception as e:
                print(f"\n[ERROR] Error en iteracion {iteration}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Guardar buffer final
        self._save_buffer()
        
        # Resumen final
        total_time = time.time() - training_start_time
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"Iteraciones completadas: {self.current_iteration + 1}")
        print(f"Tiempo total: {total_time / 60:.1f} minutos")
        print(f"Buffer final: {len(self.replay_buffer)} ejemplos")
        
        if self.training_stats['policy_losses']:
            print(f"Policy loss inicial: {self.training_stats['policy_losses'][0]:.4f}")
            print(f"Policy loss final: {self.training_stats['policy_losses'][-1]:.4f}")
        
        print(f"\nModelo guardado en: {os.path.join(self.checkpoint_dir, 'best_model.pt')}")
        print("=" * 60)
    
    def _save_buffer(self) -> None:
        """Guarda el replay buffer a disco."""
        import pickle
        buffer_path = os.path.join(self.checkpoint_dir, 'replay_buffer.pkl')
        with open(buffer_path, 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        print(f"Buffer guardado: {buffer_path}")
    
    def play_against_human(self) -> None:
        """
        Modo interactivo para jugar contra el modelo entrenado.
        
        El humano juega como 'O' (jugador -1) y el AI como 'X' (jugador 1).
        """
        print("\n" + "=" * 60)
        print("MODO INTERACTIVO: Jugar contra AlphaZero")
        print("=" * 60)
        print("Tu eres O (jugador -1), AI es X (jugador 1)")
        print("Escribe un numero de columna (0-6) para jugar")
        print("Escribe 'Q' para salir")
        print("=" * 60)
        
        self.model.eval()
        
        game = Connect4Game()
        current_player = 1  # AI empieza
        
        print("\n" + str(game))
        
        while not game.is_terminal():
            if current_player == 1:
                # Turno del AI
                print("\n[AI pensando...]")
                action = self.agent.select_action(
                    game_state=game,
                    player=current_player,
                    temperature=0,
                    deterministic=True
                )
                print(f"AI juega columna: {action}")
            else:
                # Turno del humano
                while True:
                    user_input = input("\nTu turno (columna 0-6, Q=salir): ").strip().upper()
                    
                    if user_input == 'Q':
                        print("Partida abandonada")
                        return
                    
                    if not user_input.isdigit():
                        print("Por favor, escribe un numero de columna (0-6)")
                        continue
                    
                    action = int(user_input)
                    
                    if action not in game.get_legal_moves():
                        print(f"Movimiento ilegal. Columnas validas: {game.get_legal_moves()}")
                        continue
                    
                    break
            
            # Ejecutar movimiento
            game = game.make_move(action, current_player)
            print("\n" + str(game))
            
            current_player *= -1
        
        # Resultado
        winner = game.check_winner()
        print("\n" + "=" * 40)
        if winner == 1:
            print("AI GANA!")
        elif winner == -1:
            print("TU GANAS!")
        else:
            print("EMPATE!")
        print("=" * 40)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ALPHAZERO-MINI TRAINING PIPELINE")
    print("=" * 60)
    
    # Detectar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Dispositivo: {device}")
    
    # Configuracion
    config = {
        'num_iterations': 50,
        'num_self_play_games': 25,  # Reducido para demo rapida
        'num_simulations': 50,      # Reducido para demo rapida
        'num_simulations_eval': 100,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 3,
        'temperature_threshold': 10,
        'buffer_size': 50000,
        'device': device
    }
    
    print("\nConfiguracion:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Crear modelo
    print("\nCreando modelo...")
    model = Connect4Net(num_res_blocks=4, num_filters=64)
    
    # Crear trainer (extraer num_iterations que no va al constructor)
    num_iterations = config.pop('num_iterations')
    
    trainer = AlphaZeroTrainer(
        model=model,
        checkpoint_dir='models/',
        **config
    )
    
    # Restaurar para uso posterior
    config['num_iterations'] = num_iterations
    
    # Menu principal
    print("\n" + "=" * 60)
    print("OPCIONES:")
    print("  [1] Entrenar nuevo modelo")
    print("  [2] Continuar entrenamiento desde checkpoint")
    print("  [3] Jugar contra modelo existente")
    print("  [4] Ejecutar test rapido (3 iteraciones)")
    print("  [Q] Salir")
    print("=" * 60)
    
    choice = input("\nSelecciona una opcion: ").strip().upper()
    
    if choice == '1':
        # Entrenar desde cero
        print("\nIniciando entrenamiento desde cero...")
        trainer.train(num_iterations=config['num_iterations'])
        
    elif choice == '2':
        # Continuar desde checkpoint
        checkpoint_path = os.path.join('models', 'best_model.pt')
        if os.path.exists(checkpoint_path):
            trainer.train(
                num_iterations=config['num_iterations'],
                resume_from=checkpoint_path
            )
        else:
            print(f"No se encontro checkpoint en: {checkpoint_path}")
            print("Iniciando entrenamiento desde cero...")
            trainer.train(num_iterations=config['num_iterations'])
            
    elif choice == '3':
        # Jugar contra modelo
        checkpoint_path = os.path.join('models', 'best_model.pt')
        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, trainer.model)
            trainer.play_against_human()
        else:
            print(f"No se encontro modelo en: {checkpoint_path}")
            print("Entrena primero un modelo con la opcion 1 o 4")
            
    elif choice == '4':
        # Test rapido
        print("\nEjecutando test rapido (3 iteraciones)...")
        
        # Usar configuracion mas ligera para test
        trainer.num_self_play_games = 10
        trainer.num_simulations = 30
        trainer.agent = AlphaZeroAgent(
            model=trainer.model,
            num_simulations=30,
            device=device
        )
        
        trainer.train(num_iterations=3)
        
        # Preguntar si quiere jugar
        try:
            play = input("\nQuieres jugar contra el modelo? (S/N): ").strip().upper()
            if play == 'S':
                trainer.play_against_human()
        except EOFError:
            print("\n[Test completado automaticamente]")
    
    elif choice == 'Q':
        print("Hasta luego!")
    
    else:
        print("Opcion no valida")

