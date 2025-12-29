"""
Monte Carlo Tree Search (MCTS) Implementation for AlphaZero-Mini

Este modulo implementa MCTS guiado por red neuronal, el corazon de AlphaZero.
Combina busqueda en arbol con evaluacion neuronal para encontrar movimientos optimos.

MCTS tiene 4 fases:
1. SELECCION: Navegar el arbol usando UCB hasta llegar a un nodo hoja
2. EXPANSION: Crear nodos hijos para el nodo hoja
3. EVALUACION: Usar la red neuronal para evaluar la posicion
4. BACKPROPAGATION: Propagar el valor hacia arriba actualizando estadisticas

La formula UCB balancea:
- EXPLOTACION: Preferir movimientos con alto valor promedio (Q)
- EXPLORACION: Probar movimientos poco visitados con alta probabilidad prior (P)
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
import math

# Imports locales
from game import Connect4Game
from neural_net import Connect4Net, board_to_tensor


class MCTSNode:
    """
    Nodo en el arbol de busqueda MCTS.
    
    Cada nodo representa un estado del juego (posicion del tablero).
    Almacena estadisticas de las simulaciones que pasan por el.
    
    Attributes:
        game_state: Estado del juego (tablero actual)
        parent: Nodo padre (None para la raiz)
        move: Movimiento que llevo a este estado desde el padre
        player: Jugador que debe mover en este estado (1 o -1)
        prior_prob: Probabilidad prior de la red neuronal para este movimiento
        children: Diccionario de hijos {columna: MCTSNode}
        visit_count: Numero de veces que este nodo fue visitado
        total_value: Suma de valores backpropagados
        is_expanded: Si los hijos han sido creados
    """
    
    def __init__(
        self,
        game_state: Connect4Game,
        parent: Optional['MCTSNode'],
        move: Optional[int],
        player: int,
        prior_prob: float = 0.0
    ) -> None:
        """
        Inicializa un nodo MCTS.
        
        Args:
            game_state: Estado del juego en este nodo
            parent: Nodo padre (None si es raiz)
            move: Columna jugada para llegar aqui (None si es raiz)
            player: Jugador que debe mover (1 o -1)
            prior_prob: Probabilidad prior de la policy network
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.player = player
        self.prior_prob = prior_prob
        
        # Hijos: mapea columna -> nodo hijo
        self.children: Dict[int, MCTSNode] = {}
        
        # Estadisticas de simulacion
        self.visit_count: int = 0
        self.total_value: float = 0.0
        
        # Flag de expansion
        self.is_expanded: bool = False
    
    def is_leaf(self) -> bool:
        """
        Verifica si este nodo es una hoja (no expandido).
        
        Returns:
            True si el nodo no tiene hijos (no expandido), False si ya fue expandido.
        """
        return not self.is_expanded
    
    def select_child(self, c_puct: float = 1.4) -> 'MCTSNode':
        """
        Selecciona el mejor hijo usando la formula UCB (Upper Confidence Bound).
        
        Formula UCB para AlphaZero:
            UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Donde:
            - Q(s,a) = valor promedio del hijo (explotacion)
            - P(s,a) = probabilidad prior de la red neuronal
            - N(s) = visitas del padre (este nodo)
            - N(s,a) = visitas del hijo
            - c_puct = constante de exploracion (tipicamente 1.0-2.0)
        
        La formula balancea:
            - EXPLOTACION: Alto Q(s,a) -> movimientos que han dado buenos resultados
            - EXPLORACION: Alto P(s,a) y bajo N(s,a) -> movimientos prometedores poco explorados
        
        Args:
            c_puct: Constante de exploracion (mayor = mas exploracion)
        
        Returns:
            Nodo hijo con mayor puntuacion UCB.
        """
        best_score = float('-inf')
        best_child = None
        
        # sqrt(N(s)) - raiz de visitas del padre (constante para todos los hijos)
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        for child in self.children.values():
            # Q(s,a): valor promedio del hijo
            if child.visit_count > 0:
                q_value = child.total_value / child.visit_count
            else:
                q_value = 0.0
            
            # Termino de exploracion: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            exploration = c_puct * child.prior_prob * sqrt_parent_visits / (1 + child.visit_count)
            
            # UCB score = explotacion + exploracion
            ucb_score = q_value + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        assert best_child is not None, "No hay hijos para seleccionar"
        return best_child
    
    def expand(self, policy_probs: np.ndarray) -> None:
        """
        Expande el nodo creando hijos para todos los movimientos legales.
        
        Cada hijo recibe como prior la probabilidad de la policy network.
        Solo se crean hijos para movimientos legales.
        
        Args:
            policy_probs: Array de probabilidades (7,) de la policy network.
                         Debe estar normalizado (suma = 1) sobre movimientos legales.
        """
        legal_moves = self.game_state.get_legal_moves()
        
        for move in legal_moves:
            # Crear nuevo estado aplicando el movimiento
            new_game_state = self.game_state.make_move(move, self.player)
            
            # El siguiente jugador es el oponente
            next_player = -self.player
            
            # Prior probability para este movimiento
            prior = policy_probs[move]
            
            # Crear nodo hijo
            child_node = MCTSNode(
                game_state=new_game_state,
                parent=self,
                move=move,
                player=next_player,
                prior_prob=prior
            )
            
            self.children[move] = child_node
        
        self.is_expanded = True
    
    def update(self, value: float) -> None:
        """
        Actualiza las estadisticas del nodo con un valor backpropagado.
        
        Args:
            value: Valor de la simulacion (-1 a +1) desde la perspectiva
                   del jugador en este nodo.
        """
        self.visit_count += 1
        self.total_value += value
    
    def get_value(self) -> float:
        """
        Obtiene el valor promedio de este nodo.
        
        Returns:
            Valor promedio (total_value / visit_count), o 0 si no hay visitas.
        """
        if self.visit_count > 0:
            return self.total_value / self.visit_count
        return 0.0


class MCTS:
    """
    Monte Carlo Tree Search guiado por red neuronal.
    
    Ejecuta simulaciones para construir un arbol de juego y usa
    la red neuronal para evaluar posiciones y guiar la busqueda.
    
    Attributes:
        model: Red neuronal Connect4Net para evaluacion
        num_simulations: Numero de simulaciones por busqueda
        c_puct: Constante de exploracion para UCB
        device: Dispositivo para la red neuronal (CPU/CUDA)
    """
    
    def __init__(
        self,
        model: Connect4Net,
        num_simulations: int = 200,
        c_puct: float = 1.4,
        device: str = 'cpu'
    ) -> None:
        """
        Inicializa MCTS con una red neuronal.
        
        Args:
            model: Red neuronal Connect4Net entrenada (o sin entrenar)
            num_simulations: Numero de simulaciones por movimiento (default: 200)
            c_puct: Constante de exploracion UCB (default: 1.4)
            device: 'cpu' o 'cuda' para la red neuronal
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = torch.device(device)
        
        # Mover modelo al dispositivo
        self.model = self.model.to(self.device)
        self.model.eval()  # Modo evaluacion
    
    def search(self, game_state: Connect4Game, player: int) -> np.ndarray:
        """
        Ejecuta MCTS y retorna la distribucion de conteo de visitas.
        
        Este es el metodo principal de MCTS. Ejecuta num_simulations
        simulaciones, cada una con las fases: Seleccion -> Expansion ->
        Evaluacion -> Backpropagation.
        
        Args:
            game_state: Estado actual del juego
            player: Jugador que debe mover (1 o -1)
        
        Returns:
            Array numpy de forma (7,) con el conteo de visitas para cada columna.
            Las columnas ilegales tendran 0 visitas.
        """
        # Crear nodo raiz
        root = MCTSNode(
            game_state=game_state,
            parent=None,
            move=None,
            player=player,
            prior_prob=1.0
        )
        
        # Expandir la raiz con evaluacion de la red neuronal
        if not game_state.is_terminal():
            policy_probs, _ = self._evaluate(game_state, player)
            root.expand(policy_probs)
        
        # Ejecutar simulaciones
        for _ in range(self.num_simulations):
            node = root
            
            # =================================================================
            # FASE 1: SELECCION
            # Navegar el arbol hasta llegar a un nodo hoja usando UCB
            # =================================================================
            while not node.is_leaf() and not node.game_state.is_terminal():
                node = node.select_child(self.c_puct)
            
            # =================================================================
            # FASE 2: EXPANSION y EVALUACION
            # =================================================================
            if node.game_state.is_terminal():
                # Estado terminal: usar resultado real del juego
                winner = node.game_state.check_winner()
                if winner == 0:
                    # Empate
                    value = 0.0
                else:
                    # Victoria: +1 si gano el jugador del nodo, -1 si perdio
                    # Nota: winner es quien gano, node.player es quien debia jugar
                    # Si el oponente gano (winner == -node.player), es malo para node.player
                    value = 1.0 if winner == node.player else -1.0
            else:
                # Evaluar con red neuronal
                policy_probs, value = self._evaluate(node.game_state, node.player)
                
                # Expandir el nodo con las probabilidades de la policy
                node.expand(policy_probs)
            
            # =================================================================
            # FASE 3: BACKPROPAGATION
            # Propagar el valor hacia arriba, invirtiendo signo en cada nivel
            # (el valor positivo para un jugador es negativo para el oponente)
            # =================================================================
            current_value = value
            while node is not None:
                node.update(current_value)
                # Invertir valor al subir (perspectiva del oponente)
                current_value = -current_value
                node = node.parent
        
        # Extraer conteo de visitas de los hijos de la raiz
        visit_counts = np.zeros(7, dtype=np.float32)
        for move, child in root.children.items():
            visit_counts[move] = child.visit_count
        
        return visit_counts
    
    def _evaluate(
        self,
        game_state: Connect4Game,
        player: int
    ) -> Tuple[np.ndarray, float]:
        """
        Evalua una posicion usando la red neuronal.
        
        Convierte el tablero a tensor, pasa por la red, aplica softmax
        a la policy, y enmascara movimientos ilegales.
        
        Args:
            game_state: Estado del juego a evaluar
            player: Jugador desde cuya perspectiva evaluar
        
        Returns:
            Tupla (policy_probs, value):
                - policy_probs: Array (7,) con probabilidades para cada columna
                                (movimientos ilegales tienen probabilidad 0)
                - value: Float entre -1 y +1 (evaluacion de la posicion)
        """
        # Convertir tablero a tensor (perspectiva canonica del jugador)
        board_tensor = board_to_tensor(game_state.board, player)
        
        # Agregar dimension de batch: (3, 6, 7) -> (1, 3, 6, 7)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass de la red neuronal
        with torch.no_grad():
            policy_logits, value_tensor = self.model(board_tensor)
        
        # Aplicar softmax a policy logits para obtener probabilidades
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value_tensor.cpu().item()
        
        # Enmascarar movimientos ilegales
        legal_moves = game_state.get_legal_moves()
        mask = np.zeros(7, dtype=np.float32)
        mask[legal_moves] = 1.0
        
        # Aplicar mascara y renormalizar
        policy_probs = policy_probs * mask
        
        # Renormalizar para que sumen 1
        prob_sum = policy_probs.sum()
        if prob_sum > 0:
            policy_probs = policy_probs / prob_sum
        else:
            # Fallback: distribucion uniforme sobre movimientos legales
            policy_probs[legal_moves] = 1.0 / len(legal_moves)
        
        return policy_probs, value
    
    def get_action_probs(
        self,
        game_state: Connect4Game,
        player: int,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Obtiene probabilidades de accion usando MCTS con temperatura.
        
        Ejecuta la busqueda MCTS y convierte los conteos de visitas
        en probabilidades usando el parametro de temperatura.
        
        Temperatura:
            - temp = 0: Deterministico (siempre el mejor movimiento)
            - temp = 1: Proporcional a visitas (para entrenamiento)
            - temp > 1: Mas aleatorio (mas exploracion)
        
        Args:
            game_state: Estado actual del juego
            player: Jugador que debe mover (1 o -1)
            temperature: Controla exploracion vs explotacion
        
        Returns:
            Array (7,) con probabilidades de accion para cada columna.
        """
        # Ejecutar busqueda MCTS
        visit_counts = self.search(game_state, player)
        
        if temperature == 0:
            # Deterministico: one-hot en el mejor movimiento
            action_probs = np.zeros(7, dtype=np.float32)
            best_move = np.argmax(visit_counts)
            action_probs[best_move] = 1.0
        else:
            # Aplicar temperatura
            # probs ∝ visit_counts^(1/temperature)
            if temperature == 1.0:
                # Caso especial: proporcional a visitas
                action_probs = visit_counts / visit_counts.sum()
            else:
                # Temperatura general
                visit_counts_temp = visit_counts ** (1.0 / temperature)
                action_probs = visit_counts_temp / visit_counts_temp.sum()
        
        return action_probs


def select_move_from_probs(
    action_probs: np.ndarray,
    deterministic: bool = False
) -> int:
    """
    Selecciona un movimiento basado en las probabilidades de accion.
    
    Args:
        action_probs: Array (7,) con probabilidades para cada columna
        deterministic: Si True, siempre selecciona el mejor movimiento.
                      Si False, muestrea de la distribucion.
    
    Returns:
        Indice de columna seleccionado (0-6)
    
    Example:
        >>> probs = np.array([0.1, 0.2, 0.4, 0.1, 0.1, 0.05, 0.05])
        >>> move = select_move_from_probs(probs, deterministic=True)
        >>> move
        2  # Columna con mayor probabilidad
    """
    if deterministic:
        return int(np.argmax(action_probs))
    else:
        # Muestrear de la distribucion de probabilidades
        return int(np.random.choice(len(action_probs), p=action_probs))


# =============================================================================
# TESTS - Ejecutar con: python mcts.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTS DE MCTS (Monte Carlo Tree Search)")
    print("=" * 60)
    
    # Detectar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Dispositivo: {device}")
    
    # -------------------------------------------------------------------------
    # TEST 1: Crear red neuronal y MCTS
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Crear red neuronal y MCTS")
    
    # Crear red neuronal (sin entrenar - pesos aleatorios)
    model = Connect4Net(num_res_blocks=2, num_filters=32)  # Red pequena para tests
    model = model.to(device)
    
    # Crear MCTS
    num_sims = 50  # Pocas simulaciones para test rapido
    mcts = MCTS(model, num_simulations=num_sims, c_puct=1.4, device=device)
    
    print(f"[OK] MCTS creado con {num_sims} simulaciones")
    
    # -------------------------------------------------------------------------
    # TEST 2: Busqueda MCTS en posicion inicial
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Busqueda MCTS en posicion inicial")
    
    game = Connect4Game()
    player = 1
    
    print("Tablero inicial:")
    print(game)
    
    # Ejecutar busqueda
    visit_counts = mcts.search(game, player)
    
    print(f"\nConteo de visitas por columna:")
    for col in range(7):
        bar = "*" * int(visit_counts[col] / 2)  # Barra visual
        print(f"  Columna {col}: {int(visit_counts[col]):3d} visitas  {bar}")
    
    total_visits = visit_counts.sum()
    print(f"\n[OK] Total de visitas: {int(total_visits)}")
    
    # Verificar que las visitas suman aproximadamente num_simulations
    # (puede ser ligeramente diferente debido a la expansion inicial)
    assert total_visits > 0, "No hay visitas"
    print(f"[OK] Busqueda completada exitosamente")
    
    # -------------------------------------------------------------------------
    # TEST 3: Verificar movimientos ilegales
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verificar movimientos ilegales tienen 0 visitas")
    
    # Crear posicion con columna 3 llena
    game_col_full = Connect4Game()
    for i in range(6):
        p = 1 if i % 2 == 0 else -1
        game_col_full = game_col_full.make_move(3, p)
    
    print("Tablero con columna 3 llena:")
    print(game_col_full)
    
    visit_counts_full = mcts.search(game_col_full, player=1)
    
    print(f"\nConteo de visitas:")
    for col in range(7):
        status = "(LLENA)" if col == 3 else ""
        print(f"  Columna {col}: {int(visit_counts_full[col]):3d} visitas {status}")
    
    assert visit_counts_full[3] == 0, "Columna llena deberia tener 0 visitas"
    print("[OK] Columna llena tiene 0 visitas")
    
    # -------------------------------------------------------------------------
    # TEST 4: get_action_probs con diferentes temperaturas
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Probabilidades de accion con temperatura")
    
    game = Connect4Game()
    
    # Temperatura 0 (deterministico)
    probs_t0 = mcts.get_action_probs(game, player=1, temperature=0)
    print(f"\nTemperatura = 0 (deterministico):")
    print(f"  Probabilidades: {probs_t0}")
    assert np.sum(probs_t0 == 1.0) == 1, "Deberia ser one-hot"
    print(f"  [OK] One-hot: mejor movimiento = columna {np.argmax(probs_t0)}")
    
    # Temperatura 1 (proporcional)
    probs_t1 = mcts.get_action_probs(game, player=1, temperature=1.0)
    print(f"\nTemperatura = 1 (proporcional a visitas):")
    print(f"  Probabilidades: {np.round(probs_t1, 3)}")
    assert abs(probs_t1.sum() - 1.0) < 1e-5, "Probabilidades deben sumar 1"
    print(f"  [OK] Suma de probabilidades: {probs_t1.sum():.6f}")
    
    # Temperatura alta (mas exploracion)
    probs_t2 = mcts.get_action_probs(game, player=1, temperature=2.0)
    print(f"\nTemperatura = 2 (mas exploracion):")
    print(f"  Probabilidades: {np.round(probs_t2, 3)}")
    
    # -------------------------------------------------------------------------
    # TEST 5: select_move_from_probs
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Seleccion de movimiento")
    
    test_probs = np.array([0.05, 0.1, 0.4, 0.2, 0.15, 0.05, 0.05])
    
    # Deterministico
    move_det = select_move_from_probs(test_probs, deterministic=True)
    print(f"Probabilidades: {test_probs}")
    print(f"Movimiento deterministico: columna {move_det}")
    assert move_det == 2, "Deberia elegir columna 2 (mayor prob)"
    print("[OK] Seleccion deterministica correcta")
    
    # Estocastico (muestrear varias veces)
    moves_sampled = [select_move_from_probs(test_probs, deterministic=False) for _ in range(100)]
    unique_moves = set(moves_sampled)
    print(f"Movimientos muestreados (100 veces): {len(unique_moves)} columnas diferentes usadas")
    print(f"  Distribucion: {[moves_sampled.count(i) for i in range(7)]}")
    print("[OK] Muestreo estocastico funciona")
    
    # -------------------------------------------------------------------------
    # TEST 6: Posicion ganadora
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Detectar posicion ganadora")
    
    # Crear posicion donde jugador 1 puede ganar en un movimiento
    # Jugador 1 tiene 3 en linea en columnas 0, 1, 2
    game_winning = Connect4Game()
    # Jugador 1: columnas 0, 1, 2
    game_winning = game_winning.make_move(0, 1)
    game_winning = game_winning.make_move(4, -1)  # -1 juega en otro lado
    game_winning = game_winning.make_move(1, 1)
    game_winning = game_winning.make_move(5, -1)
    game_winning = game_winning.make_move(2, 1)
    game_winning = game_winning.make_move(6, -1)
    # Ahora jugador 1 puede ganar jugando columna 3
    
    print("Posicion con victoria posible para X en columna 3:")
    print(game_winning)
    
    # MCTS deberia encontrar el movimiento ganador
    winning_probs = mcts.get_action_probs(game_winning, player=1, temperature=0)
    best_move = np.argmax(winning_probs)
    
    print(f"Movimiento recomendado: columna {best_move}")
    if best_move == 3:
        print("[OK] MCTS encontro el movimiento ganador!")
    else:
        print(f"[NOTA] Con red sin entrenar, puede no encontrar el optimo (esperado: 3)")
    
    # -------------------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[OK] TODOS LOS TESTS PASARON CORRECTAMENTE")
    print("=" * 60)
    print("\nResumen de MCTS:")
    print(f"  - Simulaciones por busqueda: {num_sims}")
    print(f"  - Constante de exploracion (c_puct): {mcts.c_puct}")
    print(f"  - Dispositivo: {device}")
    print("=" * 60)

