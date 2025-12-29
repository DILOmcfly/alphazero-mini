"""
Connect4 Game Implementation for AlphaZero-Mini

Este módulo implementa la lógica completa del juego Connect4 (4 en línea).
El tablero es de 6 filas x 7 columnas, donde las fichas "caen" por gravedad.

Convenciones:
- Jugador 1: fichas con valor 1
- Jugador -1: fichas con valor -1  
- Celdas vacías: valor 0
"""

import numpy as np
from typing import List, Optional, Tuple
import copy


class Connect4Game:
    """
    Implementación del juego Connect4 (4 en línea).
    
    El tablero se representa como una matriz numpy de 6x7 donde:
    - La fila 0 es la parte SUPERIOR del tablero (donde aparecen las fichas primero visualmente,
      pero donde caen al final por gravedad)
    - La fila 5 es la parte INFERIOR del tablero (donde caen las fichas primero)
    - Las columnas van de 0 (izquierda) a 6 (derecha)
    
    Attributes:
        rows (int): Número de filas del tablero (6)
        cols (int): Número de columnas del tablero (7)
        board (np.ndarray): Matriz que representa el estado del tablero
    """
    
    # Constantes del juego
    ROWS: int = 6
    COLS: int = 7
    WIN_LENGTH: int = 4  # Fichas consecutivas para ganar
    
    def __init__(self, board: Optional[np.ndarray] = None) -> None:
        """
        Inicializa un nuevo juego de Connect4.
        
        Args:
            board: Estado inicial del tablero (opcional). Si es None,
                   se crea un tablero vacío lleno de ceros.
        """
        if board is not None:
            self.board = board.copy()
        else:
            # Tablero vacío: matriz de 6x7 con ceros
            self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
    
    def get_legal_moves(self) -> List[int]:
        """
        Obtiene la lista de movimientos legales (columnas disponibles).
        
        Un movimiento es legal si la columna no está llena, es decir,
        si la fila superior (fila 0) de esa columna está vacía.
        
        Returns:
            Lista de índices de columnas (0-6) donde se puede jugar.
        """
        # Una columna está disponible si su celda superior (fila 0) está vacía
        legal_moves = []
        for col in range(self.COLS):
            if self.board[0, col] == 0:
                legal_moves.append(col)
        return legal_moves
    
    def make_move(self, column: int, player: int) -> 'Connect4Game':
        """
        Realiza un movimiento en el tablero.
        
        La ficha "cae" por gravedad hasta la posición más baja disponible
        en la columna especificada.
        
        Args:
            column: Índice de la columna donde jugar (0-6)
            player: Jugador que realiza el movimiento (1 o -1)
            
        Returns:
            Nuevo objeto Connect4Game con el movimiento aplicado.
            (No modifica el estado actual, devuelve una copia)
            
        Raises:
            ValueError: Si la columna es inválida o está llena.
        """
        # Validar número de columna
        if column < 0 or column >= self.COLS:
            raise ValueError(
                f"Columna inválida: {column}. Debe estar entre 0 y {self.COLS - 1}."
            )
        
        # Validar que la columna no esté llena
        if self.board[0, column] != 0:
            raise ValueError(
                f"Columna {column} está llena. Movimiento ilegal."
            )
        
        # Validar jugador
        if player not in [1, -1]:
            raise ValueError(
                f"Jugador inválido: {player}. Debe ser 1 o -1."
            )
        
        # Crear copia del tablero para no modificar el original
        new_board = self.board.copy()
        
        # Encontrar la fila más baja disponible (gravedad)
        # Empezamos desde abajo (fila 5) y subimos hasta encontrar una celda vacía
        for row in range(self.ROWS - 1, -1, -1):
            if new_board[row, column] == 0:
                new_board[row, column] = player
                break
        
        return Connect4Game(new_board)
    
    def check_winner(self) -> Optional[int]:
        """
        Verifica si hay un ganador o empate.
        
        Revisa todas las posibles líneas de 4 fichas consecutivas:
        - Horizontales (→)
        - Verticales (↓)
        - Diagonales descendentes (↘)
        - Diagonales ascendentes (↗)
        
        Returns:
            1: Si el jugador 1 gana
            -1: Si el jugador -1 gana
            0: Si hay empate (tablero lleno sin ganador)
            None: Si el juego continúa (no hay ganador ni empate)
        """
        # =====================================================
        # VERIFICACIÓN DE VICTORIA HORIZONTAL (→)
        # Para cada fila, revisamos ventanas de 4 columnas consecutivas
        # =====================================================
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):  # -3 porque necesitamos 4 espacios
                window = self.board[row, col:col + 4]
                winner = self._check_window(window)
                if winner is not None:
                    return winner
        
        # =====================================================
        # VERIFICACIÓN DE VICTORIA VERTICAL (↓)
        # Para cada columna, revisamos ventanas de 4 filas consecutivas
        # =====================================================
        for row in range(self.ROWS - 3):  # -3 porque necesitamos 4 espacios
            for col in range(self.COLS):
                window = self.board[row:row + 4, col]
                winner = self._check_window(window)
                if winner is not None:
                    return winner
        
        # =====================================================
        # VERIFICACIÓN DE VICTORIA DIAGONAL DESCENDENTE (↘)
        # Desde cada posición válida, revisamos 4 celdas en diagonal
        # hacia abajo-derecha: (row, col), (row+1, col+1), (row+2, col+2), (row+3, col+3)
        # =====================================================
        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                window = np.array([
                    self.board[row + i, col + i] for i in range(4)
                ])
                winner = self._check_window(window)
                if winner is not None:
                    return winner
        
        # =====================================================
        # VERIFICACIÓN DE VICTORIA DIAGONAL ASCENDENTE (↗)
        # Desde cada posición válida, revisamos 4 celdas en diagonal
        # hacia arriba-derecha: (row, col), (row-1, col+1), (row-2, col+2), (row-3, col+3)
        # Empezamos desde fila 3 para tener espacio hacia arriba
        # =====================================================
        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                window = np.array([
                    self.board[row - i, col + i] for i in range(4)
                ])
                winner = self._check_window(window)
                if winner is not None:
                    return winner
        
        # =====================================================
        # VERIFICACIÓN DE EMPATE
        # Si no hay ganador y el tablero está lleno (no hay movimientos legales)
        # =====================================================
        if len(self.get_legal_moves()) == 0:
            return 0  # Empate
        
        # El juego continúa
        return None
    
    def _check_window(self, window: np.ndarray) -> Optional[int]:
        """
        Verifica si una ventana de 4 celdas contiene un ganador.
        
        Args:
            window: Array de 4 elementos representando fichas consecutivas.
            
        Returns:
            1 si jugador 1 gana, -1 si jugador -1 gana, None si no hay ganador.
        """
        # Si todas las 4 fichas son del jugador 1
        if np.all(window == 1):
            return 1
        # Si todas las 4 fichas son del jugador -1
        if np.all(window == -1):
            return -1
        return None
    
    def is_terminal(self) -> bool:
        """
        Verifica si el juego ha terminado.
        
        El juego termina cuando hay un ganador o cuando hay empate
        (tablero lleno sin ganador).
        
        Returns:
            True si el juego terminó, False si continúa.
        """
        return self.check_winner() is not None
    
    def get_board_copy(self) -> np.ndarray:
        """
        Obtiene una copia profunda del tablero actual.
        
        Útil para operaciones que necesitan modificar el tablero
        sin afectar el estado del juego.
        
        Returns:
            Copia del tablero como numpy array.
        """
        return self.board.copy()
    
    def get_canonical_board(self, player: int) -> np.ndarray:
        """
        Obtiene el tablero desde la perspectiva del jugador actual.
        
        Esto es importante para la red neuronal: siempre ve el tablero
        como si fuera el jugador 1. Si el jugador actual es -1,
        invertimos los signos de todas las fichas.
        
        Ejemplo:
            Si player=1: devuelve el tablero tal cual
            Si player=-1: fichas 1 se vuelven -1 y viceversa
        
        Args:
            player: Jugador actual (1 o -1)
            
        Returns:
            Tablero transformado donde el jugador actual siempre es "1".
        """
        return self.board * player
    
    def __str__(self) -> str:
        """
        Representación visual del tablero para debugging.
        
        Muestra el tablero con:
        - 'X' para el jugador 1
        - 'O' para el jugador -1
        - '.' para celdas vacías
        - Números de columna en la parte inferior
        
        Returns:
            String con la representación visual del tablero.
        """
        # Mapeo de valores a símbolos
        symbols = {0: '.', 1: 'X', -1: 'O'}
        
        lines = []
        lines.append("+" + "---+" * self.COLS)
        
        for row in range(self.ROWS):
            row_str = "|"
            for col in range(self.COLS):
                symbol = symbols[self.board[row, col]]
                row_str += f" {symbol} |"
            lines.append(row_str)
            lines.append("+" + "---+" * self.COLS)
        
        # Números de columna
        col_numbers = "  " + "   ".join(str(i) for i in range(self.COLS)) + "  "
        lines.append(col_numbers)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Representación para debugging."""
        return f"Connect4Game(board=\n{self.board})"


# =============================================================================
# TESTS - Ejecutar con: python game.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTS DE CONNECT4")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # TEST 1: Creación del juego y movimientos básicos
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Creacion del juego y movimientos basicos")
    game = Connect4Game()
    print("Tablero inicial:")
    print(game)
    
    # Verificar que hay 7 movimientos legales al inicio
    legal_moves = game.get_legal_moves()
    assert legal_moves == [0, 1, 2, 3, 4, 5, 6], "Error: Deberían haber 7 movimientos legales"
    print(f"[OK] Movimientos legales iniciales: {legal_moves}")
    
    # Hacer algunos movimientos
    game = game.make_move(3, 1)   # Jugador 1 juega en columna 3
    game = game.make_move(3, -1)  # Jugador -1 juega en columna 3
    game = game.make_move(4, 1)   # Jugador 1 juega en columna 4
    print("\nDespués de 3 movimientos:")
    print(game)
    print("[OK] Movimientos básicos funcionan correctamente")
    
    # -------------------------------------------------------------------------
    # TEST 2: Victoria horizontal
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Victoria horizontal")
    game = Connect4Game()
    # Jugador 1 hace 4 en línea horizontal
    game = game.make_move(0, 1)
    game = game.make_move(0, -1)  # -1 juega arriba
    game = game.make_move(1, 1)
    game = game.make_move(1, -1)
    game = game.make_move(2, 1)
    game = game.make_move(2, -1)
    game = game.make_move(3, 1)   # ¡4 en línea!
    print(game)
    winner = game.check_winner()
    assert winner == 1, f"Error: Debería ganar jugador 1, pero ganó {winner}"
    assert game.is_terminal() == True, "Error: El juego debería haber terminado"
    print(f"[OK] Ganador detectado correctamente: Jugador {winner}")
    
    # -------------------------------------------------------------------------
    # TEST 3: Victoria vertical
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Victoria vertical")
    game = Connect4Game()
    # Jugador 1 hace 4 en línea vertical en columna 0
    for i in range(4):
        game = game.make_move(0, 1)
        if i < 3:  # -1 juega en otra columna
            game = game.make_move(1, -1)
    print(game)
    winner = game.check_winner()
    assert winner == 1, f"Error: Debería ganar jugador 1, pero ganó {winner}"
    print(f"[OK] Victoria vertical detectada: Jugador {winner}")
    
    # -------------------------------------------------------------------------
    # TEST 4: Victoria diagonal (↘)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Victoria diagonal descendente")
    game = Connect4Game()
    # Construir una diagonal para jugador 1
    # Col 0: solo 1 ficha de jugador 1
    game = game.make_move(0, 1)
    # Col 1: 1 ficha de -1, luego 1 de jugador 1
    game = game.make_move(1, -1)
    game = game.make_move(1, 1)
    # Col 2: 2 fichas de -1, luego 1 de jugador 1
    game = game.make_move(2, -1)
    game = game.make_move(2, -1)
    game = game.make_move(2, 1)
    # Col 3: 3 fichas de -1, luego 1 de jugador 1
    game = game.make_move(3, -1)
    game = game.make_move(3, -1)
    game = game.make_move(3, -1)
    game = game.make_move(3, 1)  # ¡Diagonal completa!
    print(game)
    winner = game.check_winner()
    assert winner == 1, f"Error: Debería ganar jugador 1, pero ganó {winner}"
    print(f"[OK] Victoria diagonal \\ detectada: Jugador {winner}")
    
    # -------------------------------------------------------------------------
    # TEST 5: Victoria diagonal (↗)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Victoria diagonal ascendente")
    game = Connect4Game()
    # Construir una diagonal ascendente para jugador -1
    # Col 3: solo 1 ficha de -1
    game = game.make_move(3, -1)
    # Col 2: 1 ficha de 1, luego 1 de -1
    game = game.make_move(2, 1)
    game = game.make_move(2, -1)
    # Col 1: 2 fichas de 1, luego 1 de -1
    game = game.make_move(1, 1)
    game = game.make_move(1, 1)
    game = game.make_move(1, -1)
    # Col 0: 3 fichas de 1, luego 1 de -1
    game = game.make_move(0, 1)
    game = game.make_move(0, 1)
    game = game.make_move(0, 1)
    game = game.make_move(0, -1)  # ¡Diagonal completa!
    print(game)
    winner = game.check_winner()
    assert winner == -1, f"Error: Debería ganar jugador -1, pero ganó {winner}"
    print(f"[OK] Victoria diagonal / detectada: Jugador {winner}")
    
    # -------------------------------------------------------------------------
    # TEST 6: Columna llena
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Columna llena")
    game = Connect4Game()
    # Llenar la columna 0
    for i in range(6):
        player = 1 if i % 2 == 0 else -1
        game = game.make_move(0, player)
    print(game)
    legal_moves = game.get_legal_moves()
    assert 0 not in legal_moves, "Error: Columna 0 debería estar llena"
    print(f"[OK] Columna llena detectada. Movimientos legales: {legal_moves}")
    
    # Intentar jugar en columna llena debe lanzar excepción
    try:
        game.make_move(0, 1)
        print("[ERROR] Error: Debería haber lanzado excepción")
    except ValueError as e:
        print(f"[OK] Excepción correcta: {e}")
    
    # -------------------------------------------------------------------------
    # TEST 7: Tablero canónico
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Tablero canonico")
    game = Connect4Game()
    game = game.make_move(3, 1)
    game = game.make_move(4, -1)
    
    canonical_p1 = game.get_canonical_board(1)
    canonical_p2 = game.get_canonical_board(-1)
    
    print("Tablero original:")
    print(game.board)
    print("\nDesde perspectiva de jugador 1:")
    print(canonical_p1)
    print("\nDesde perspectiva de jugador -1:")
    print(canonical_p2)
    
    assert np.array_equal(canonical_p1, game.board), "Error en tablero canónico P1"
    assert np.array_equal(canonical_p2, -game.board), "Error en tablero canónico P2"
    print("[OK] Tablero canónico funciona correctamente")
    
    # -------------------------------------------------------------------------
    # TEST 8: Partida completa simulada
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Partida completa simulada (movimientos aleatorios)")
    import random
    
    game = Connect4Game()
    current_player = 1
    move_count = 0
    
    while not game.is_terminal():
        legal_moves = game.get_legal_moves()
        move = random.choice(legal_moves)
        game = game.make_move(move, current_player)
        current_player *= -1  # Cambiar turno
        move_count += 1
    
    print(f"Partida terminada en {move_count} movimientos:")
    print(game)
    winner = game.check_winner()
    if winner == 0:
        print("[OK] Resultado: Empate")
    else:
        print(f"[OK] Ganador: Jugador {winner}")
    
    print("\n" + "=" * 60)
    print("[OK] TODOS LOS TESTS PASARON CORRECTAMENTE")
    print("=" * 60)
    
    # =========================================================================
    # MODO INTERACTIVO - Jugar contra ti mismo
    # =========================================================================
    print("\n")
    print("=" * 60)
    print("       CONNECT4 - MODO INTERACTIVO")
    print("=" * 60)
    print("\nQuieres jugar una partida interactiva?")
    print("  [S] Si, quiero jugar")
    print("  [N] No, salir")
    
    choice = input("\nTu eleccion: ").strip().upper()
    
    if choice == 'S':
        print("\n" + "-" * 60)
        print("INSTRUCCIONES:")
        print("-" * 60)
        print("- Jugador X (1)  va primero")
        print("- Jugador O (-1) va segundo")
        print("- Escribe un numero de columna (0-6) para jugar")
        print("- Escribe 'Q' para abandonar la partida")
        print("- Escribe 'R' para reiniciar la partida")
        print("-" * 60)
        
        def play_interactive_game():
            """Ejecuta una partida interactiva de Connect4."""
            game = Connect4Game()
            current_player = 1
            player_symbols = {1: 'X', -1: 'O'}
            
            print("\n*** NUEVA PARTIDA ***\n")
            print(game)
            
            while not game.is_terminal():
                symbol = player_symbols[current_player]
                print(f"\n>> Turno del Jugador {symbol} (jugador {current_player})")
                print(f"   Columnas disponibles: {game.get_legal_moves()}")
                
                # Obtener input del usuario
                user_input = input("   Tu movimiento (0-6, Q=salir, R=reiniciar): ").strip().upper()
                
                # Manejar comandos especiales
                if user_input == 'Q':
                    print("\n*** Partida abandonada ***")
                    return False  # No reiniciar
                
                if user_input == 'R':
                    print("\n*** Reiniciando partida... ***")
                    return True  # Reiniciar
                
                # Validar que sea un numero
                if not user_input.isdigit():
                    print("   [!] Error: Escribe un numero de columna (0-6)")
                    continue
                
                column = int(user_input)
                
                # Validar rango de columna
                if column < 0 or column > 6:
                    print("   [!] Error: La columna debe estar entre 0 y 6")
                    continue
                
                # Validar que la columna no este llena
                if column not in game.get_legal_moves():
                    print(f"   [!] Error: La columna {column} esta llena")
                    continue
                
                # Realizar el movimiento
                game = game.make_move(column, current_player)
                
                # Limpiar pantalla simulado (lineas en blanco)
                print("\n" * 2)
                print(game)
                
                # Cambiar turno
                current_player *= -1
            
            # Juego terminado - mostrar resultado
            print("\n" + "=" * 40)
            print("         JUEGO TERMINADO!")
            print("=" * 40)
            
            winner = game.check_winner()
            if winner == 0:
                print("\n   Resultado: EMPATE!")
            else:
                symbol = player_symbols[winner]
                print(f"\n   GANADOR: Jugador {symbol} (jugador {winner})!")
            
            print("\n" + "=" * 40)
            
            # Preguntar si quiere jugar de nuevo
            again = input("\nQuieres jugar otra partida? (S/N): ").strip().upper()
            return again == 'S'
        
        # Loop principal de juego
        keep_playing = True
        while keep_playing:
            keep_playing = play_interactive_game()
        
        print("\nGracias por jugar Connect4!")
        print("=" * 60)
    else:
        print("\nHasta luego!")

