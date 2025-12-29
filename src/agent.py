"""
Self-Play Agent Implementation for AlphaZero-Mini

Este modulo implementa el agente que juega partidas contra si mismo
usando MCTS + Red Neuronal, y recolecta datos de entrenamiento.

El proceso de Self-Play:
1. El agente juega una partida completa contra si mismo
2. En cada turno, usa MCTS para decidir el movimiento
3. Guarda cada posicion junto con la politica MCTS
4. Al final, etiqueta cada posicion con el resultado (-1, 0, +1)
5. Aplica augmentacion por simetria (flip horizontal)

Los datos generados se usan para entrenar la red neuronal.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
from collections import deque

# Imports locales
from game import Connect4Game
from neural_net import Connect4Net, board_to_tensor
from mcts import MCTS, select_move_from_probs


def flip_board_horizontal(board: np.ndarray) -> np.ndarray:
    """
    Voltea el tablero horizontalmente (espejo izquierda-derecha).
    
    Connect4 es simetrico horizontalmente, por lo que voltear
    el tablero produce una posicion equivalente.
    
    Args:
        board: Tablero numpy de forma (6, 7)
    
    Returns:
        Tablero volteado de forma (6, 7)
    
    Example:
        >>> board = np.array([[1, 0, 0, 0, 0, 0, -1]])
        >>> flip_board_horizontal(board)
        array([[-1, 0, 0, 0, 0, 0, 1]])
    """
    # [:, ::-1] invierte el orden de las columnas
    return board[:, ::-1].copy()


def flip_policy_horizontal(policy: np.ndarray) -> np.ndarray:
    """
    Voltea las probabilidades de politica horizontalmente.
    
    Si el tablero se voltea, las columnas cambian de posicion:
    [0, 1, 2, 3, 4, 5, 6] → [6, 5, 4, 3, 2, 1, 0]
    
    Args:
        policy: Array de probabilidades de forma (7,)
    
    Returns:
        Array de probabilidades volteado de forma (7,)
    
    Example:
        >>> policy = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01])
        >>> flip_policy_horizontal(policy)
        array([0.01, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5])
    """
    return policy[::-1].copy()


class AlphaZeroAgent:
    """
    Agente de AlphaZero que juega Connect4 usando MCTS + Red Neuronal.
    
    Este agente puede:
    1. Jugar partidas contra si mismo (self-play)
    2. Recolectar datos de entrenamiento de esas partidas
    3. Seleccionar acciones para jugar contra humanos
    
    El self-play es crucial para AlphaZero:
    - Genera datos de entrenamiento diversos
    - Mejora iterativamente al jugar contra versiones mejoradas de si mismo
    
    Attributes:
        model: Red neuronal Connect4Net
        mcts: Instancia de MCTS para busqueda
        device: Dispositivo (CPU/CUDA)
    """
    
    def __init__(
        self,
        model: Connect4Net,
        num_simulations: int = 200,
        c_puct: float = 1.4,
        device: str = 'cpu'
    ) -> None:
        """
        Inicializa el agente AlphaZero.
        
        Args:
            model: Red neuronal Connect4Net (entrenada o sin entrenar)
            num_simulations: Numero de simulaciones MCTS por movimiento
            c_puct: Constante de exploracion para MCTS
            device: 'cpu' o 'cuda'
        """
        self.model = model
        self.device = torch.device(device)
        
        # Mover modelo al dispositivo
        self.model = self.model.to(self.device)
        
        # Crear instancia de MCTS
        self.mcts = MCTS(
            model=self.model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device
        )
    
    def play_game(
        self,
        temperature_threshold: int = 10
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], int]:
        """
        Juega una partida completa contra si mismo y recolecta datos.
        
        El proceso:
        1. Iniciar partida vacia
        2. En cada turno:
           - Usar MCTS para obtener probabilidades de accion
           - Guardar (tablero, politica, jugador)
           - Ejecutar movimiento
           - Cambiar de jugador
        3. Al terminar:
           - Determinar ganador
           - Etiquetar cada ejemplo con el resultado
           - Aplicar augmentacion por simetria
        
        Args:
            temperature_threshold: Primeros N movimientos usan temp=1.0
                                  Movimientos posteriores usan temp=0.1
                                  Esto promueve diversidad al inicio
        
        Returns:
            Tupla (training_examples, winner):
                - training_examples: Lista de (board, policy, value)
                  * board: np.ndarray (6, 7) en forma canonica
                  * policy: np.ndarray (7,) probabilidades MCTS
                  * value: float (-1, 0, o +1) resultado final
                - winner: 1, -1, o 0 (empate)
        """
        game = Connect4Game()
        current_player = 1
        move_count = 0
        
        # Guardar ejemplos temporales: (canonical_board, policy, player)
        temp_examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
        
        while not game.is_terminal():
            # Determinar temperatura basada en el numero de movimientos
            # Primeros movimientos: alta temperatura (exploracion)
            # Movimientos posteriores: baja temperatura (explotacion)
            if move_count < temperature_threshold:
                temperature = 1.0
            else:
                temperature = 0.1
            
            # Obtener probabilidades de accion usando MCTS
            action_probs = self.mcts.get_action_probs(
                game_state=game,
                player=current_player,
                temperature=temperature
            )
            
            # Obtener tablero canonico (desde perspectiva del jugador actual)
            canonical_board = game.get_canonical_board(current_player)
            
            # Guardar ejemplo temporal
            temp_examples.append((
                canonical_board.copy(),
                action_probs.copy(),
                current_player
            ))
            
            # Seleccionar movimiento (muestreo estocastico para diversidad)
            move = select_move_from_probs(action_probs, deterministic=False)
            
            # Ejecutar movimiento
            game = game.make_move(move, current_player)
            
            # Cambiar de jugador
            current_player *= -1
            move_count += 1
        
        # Obtener resultado final
        winner = game.check_winner()
        
        # Convertir ejemplos temporales a ejemplos de entrenamiento
        # Agregando el valor final desde la perspectiva de cada jugador
        training_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        
        for board, policy, player in temp_examples:
            # Determinar valor desde la perspectiva del jugador que hizo el movimiento
            if winner == 0:
                # Empate
                value = 0.0
            elif winner == player:
                # Este jugador gano
                value = 1.0
            else:
                # Este jugador perdio
                value = -1.0
            
            # Agregar ejemplo original
            training_examples.append((board, policy, value))
            
            # Agregar ejemplo con augmentacion por simetria (flip horizontal)
            # Connect4 es simetrico: voltear el tablero da una posicion equivalente
            flipped_board = flip_board_horizontal(board)
            flipped_policy = flip_policy_horizontal(policy)
            training_examples.append((flipped_board, flipped_policy, value))
        
        return training_examples, winner
    
    def play_games(
        self,
        num_games: int,
        temperature_threshold: int = 10
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Dict[str, int]]:
        """
        Juega multiples partidas y agrega los datos de entrenamiento.
        
        Args:
            num_games: Numero de partidas a jugar
            temperature_threshold: Umbral de temperatura para cada partida
        
        Returns:
            Tupla (all_examples, stats):
                - all_examples: Lista de todos los ejemplos de entrenamiento
                - stats: Diccionario con estadisticas
                  * 'wins_p1': Victorias del jugador 1
                  * 'wins_p2': Victorias del jugador -1
                  * 'draws': Empates
                  * 'total_moves': Total de movimientos en todas las partidas
        """
        all_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        stats = {
            'wins_p1': 0,
            'wins_p2': 0,
            'draws': 0,
            'total_moves': 0
        }
        
        print(f"Iniciando self-play: {num_games} partidas")
        print("-" * 40)
        
        for game_idx in range(num_games):
            # Jugar una partida
            examples, winner = self.play_game(temperature_threshold)
            
            # Agregar ejemplos
            all_examples.extend(examples)
            
            # Actualizar estadisticas
            # Nota: cada ejemplo original genera 2 (con flip), 
            # asi que movimientos = len(examples) / 2
            game_moves = len(examples) // 2
            stats['total_moves'] += game_moves
            
            if winner == 1:
                stats['wins_p1'] += 1
            elif winner == -1:
                stats['wins_p2'] += 1
            else:
                stats['draws'] += 1
            
            # Imprimir progreso cada 10 partidas o al final
            if (game_idx + 1) % 10 == 0 or game_idx == num_games - 1:
                games_played = game_idx + 1
                win_rate_p1 = stats['wins_p1'] / games_played * 100
                win_rate_p2 = stats['wins_p2'] / games_played * 100
                draw_rate = stats['draws'] / games_played * 100
                avg_moves = stats['total_moves'] / games_played
                
                print(f"Partidas: {games_played}/{num_games} | "
                      f"P1: {win_rate_p1:.1f}% | P2: {win_rate_p2:.1f}% | "
                      f"Empates: {draw_rate:.1f}% | Mov/partida: {avg_moves:.1f}")
        
        print("-" * 40)
        print(f"Self-play completado. Total ejemplos: {len(all_examples)}")
        
        return all_examples, stats
    
    def select_action(
        self,
        game_state: Connect4Game,
        player: int,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> int:
        """
        Selecciona una accion para un estado dado.
        
        Util para jugar contra humanos o evaluar el agente.
        
        Args:
            game_state: Estado actual del juego
            player: Jugador que debe mover (1 o -1)
            temperature: Temperatura para MCTS
            deterministic: Si True, siempre elige el mejor movimiento
        
        Returns:
            Columna seleccionada (0-6)
        """
        # Obtener probabilidades de accion
        action_probs = self.mcts.get_action_probs(
            game_state=game_state,
            player=player,
            temperature=temperature
        )
        
        # Seleccionar movimiento
        move = select_move_from_probs(action_probs, deterministic=deterministic)
        
        return move
    
    def get_action_probs_for_state(
        self,
        game_state: Connect4Game,
        player: int,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Obtiene probabilidades de accion para un estado dado.
        
        Args:
            game_state: Estado del juego
            player: Jugador actual
            temperature: Temperatura para MCTS
        
        Returns:
            Array de probabilidades (7,)
        """
        return self.mcts.get_action_probs(
            game_state=game_state,
            player=player,
            temperature=temperature
        )


# =============================================================================
# TESTS - Ejecutar con: python agent.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTS DE ALPHAZERO AGENT (Self-Play)")
    print("=" * 60)
    
    # Detectar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Dispositivo: {device}")
    
    # -------------------------------------------------------------------------
    # TEST 1: Crear agente y jugar una partida
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Crear agente y jugar una partida")
    
    # Crear red neuronal pequena para tests rapidos
    model = Connect4Net(num_res_blocks=2, num_filters=32)
    model = model.to(device)
    
    # Crear agente con pocas simulaciones para test rapido
    agent = AlphaZeroAgent(
        model=model,
        num_simulations=30,  # Pocas para test rapido
        device=device
    )
    
    print("Jugando una partida de self-play...")
    examples, winner = agent.play_game(temperature_threshold=10)
    
    print(f"\n[OK] Partida completada")
    print(f"     - Ganador: {'Jugador 1 (X)' if winner == 1 else 'Jugador -1 (O)' if winner == -1 else 'Empate'}")
    print(f"     - Ejemplos generados: {len(examples)}")
    print(f"     - Movimientos en la partida: {len(examples) // 2}")
    
    # Verificar estructura de ejemplos
    if len(examples) > 0:
        sample_board, sample_policy, sample_value = examples[0]
        print(f"\n     Ejemplo de datos de entrenamiento:")
        print(f"     - Forma del tablero: {sample_board.shape}")
        print(f"     - Forma de la politica: {sample_policy.shape}")
        print(f"     - Valor: {sample_value}")
        
        # Verificar que policy suma ~1
        policy_sum = sample_policy.sum()
        print(f"     - Suma de politica: {policy_sum:.6f}")
        assert abs(policy_sum - 1.0) < 1e-5, "Policy deberia sumar 1"
        print(f"     [OK] Politica suma 1")
        
        # Verificar que value esta en [-1, 1]
        assert sample_value in [-1.0, 0.0, 1.0], f"Valor invalido: {sample_value}"
        print(f"     [OK] Valor en rango valido")
    
    # -------------------------------------------------------------------------
    # TEST 2: Jugar multiples partidas
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Jugar multiples partidas (10 partidas)")
    
    all_examples, stats = agent.play_games(num_games=10, temperature_threshold=8)
    
    print(f"\n[OK] Self-play de 10 partidas completado")
    print(f"     - Total ejemplos: {len(all_examples)}")
    print(f"     - Victorias P1: {stats['wins_p1']}")
    print(f"     - Victorias P2: {stats['wins_p2']}")
    print(f"     - Empates: {stats['draws']}")
    print(f"     - Promedio movimientos/partida: {stats['total_moves'] / 10:.1f}")
    
    # -------------------------------------------------------------------------
    # TEST 3: Verificar augmentacion por simetria
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verificar augmentacion por simetria")
    
    # Jugar una partida y verificar que hay pares de ejemplos
    examples_single, _ = agent.play_game(temperature_threshold=5)
    num_moves = len(examples_single) // 2  # Cada movimiento genera 2 ejemplos
    
    print(f"     Movimientos en partida: {num_moves}")
    print(f"     Ejemplos generados: {len(examples_single)} (x2 por simetria)")
    
    # Verificar que los ejemplos vienen en pares (original + flip)
    if len(examples_single) >= 2:
        original_board, original_policy, original_value = examples_single[0]
        flipped_board, flipped_policy, flipped_value = examples_single[1]
        
        print(f"\n     Tablero original (columna 0 izquierda):")
        print(f"     {original_board[-1, :]}")  # Ultima fila
        
        print(f"\n     Tablero volteado (columna 6 ahora es 0):")
        print(f"     {flipped_board[-1, :]}")
        
        print(f"\n     Politica original: {np.round(original_policy, 3)}")
        print(f"     Politica volteada: {np.round(flipped_policy, 3)}")
        
        # Verificar que el flip es correcto
        expected_flip = flip_board_horizontal(original_board)
        assert np.array_equal(flipped_board, expected_flip), "Flip de tablero incorrecto"
        print(f"\n     [OK] Flip de tablero correcto")
        
        expected_policy_flip = flip_policy_horizontal(original_policy)
        assert np.allclose(flipped_policy, expected_policy_flip), "Flip de politica incorrecto"
        print(f"     [OK] Flip de politica correcto")
        
        # Valores deben ser iguales
        assert original_value == flipped_value, "Valores deben ser iguales"
        print(f"     [OK] Valores iguales: {original_value}")
    
    # -------------------------------------------------------------------------
    # TEST 4: Seleccion de accion deterministica
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Seleccion de accion deterministica")
    
    game = Connect4Game()
    
    # Seleccionar accion varias veces con deterministic=True
    actions = []
    for _ in range(5):
        action = agent.select_action(
            game_state=game,
            player=1,
            temperature=1.0,
            deterministic=True
        )
        actions.append(action)
    
    print(f"     Acciones seleccionadas (5 veces): {actions}")
    
    # Todas las acciones deben ser iguales
    assert len(set(actions)) == 1, "Acciones deterministicas deben ser iguales"
    print(f"     [OK] Todas las acciones son iguales: columna {actions[0]}")
    
    # -------------------------------------------------------------------------
    # TEST 5: Seleccion de accion estocastica
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Seleccion de accion estocastica")
    
    actions_stochastic = []
    for _ in range(20):
        action = agent.select_action(
            game_state=game,
            player=1,
            temperature=1.0,
            deterministic=False
        )
        actions_stochastic.append(action)
    
    unique_actions = set(actions_stochastic)
    print(f"     Acciones unicas en 20 intentos: {len(unique_actions)}")
    print(f"     Acciones: {actions_stochastic}")
    
    # Con temperatura 1.0, deberiamos ver variedad
    if len(unique_actions) > 1:
        print(f"     [OK] Seleccion estocastica muestra variedad")
    else:
        print(f"     [NOTA] Poca variedad (puede ser normal con red sin entrenar)")
    
    # -------------------------------------------------------------------------
    # TEST 6: Verificar que los ejemplos tienen la estructura correcta
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Verificar estructura de ejemplos de entrenamiento")
    
    # Tomar algunos ejemplos aleatorios
    if len(all_examples) >= 5:
        for idx in range(5):
            board, policy, value = all_examples[idx]
            
            # Verificar formas
            assert board.shape == (6, 7), f"Forma de tablero incorrecta: {board.shape}"
            assert policy.shape == (7,), f"Forma de politica incorrecta: {policy.shape}"
            assert isinstance(value, (int, float)), f"Tipo de valor incorrecto: {type(value)}"
            
            # Verificar rangos
            assert np.all((board >= -1) & (board <= 1)), "Valores de tablero fuera de rango"
            assert abs(policy.sum() - 1.0) < 1e-5, "Politica no suma 1"
            assert value in [-1.0, 0.0, 1.0], f"Valor fuera de rango: {value}"
    
    print(f"     [OK] Todos los ejemplos tienen estructura correcta")
    
    # -------------------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[OK] TODOS LOS TESTS PASARON CORRECTAMENTE")
    print("=" * 60)
    print("\nResumen del Agente AlphaZero:")
    print(f"  - Simulaciones MCTS: {agent.mcts.num_simulations}")
    print(f"  - Dispositivo: {device}")
    print(f"  - Self-play genera: (tablero, politica, valor)")
    print(f"  - Augmentacion: x2 por simetria horizontal")
    print("=" * 60)

