"""
Neural Network Implementation for AlphaZero-Mini

Este modulo implementa la arquitectura de red neuronal basada en ResNet
que usa AlphaZero para evaluar posiciones y sugerir movimientos.

La red tiene dos "cabezas":
1. Policy Head: Predice probabilidades para cada movimiento posible (7 columnas)
2. Value Head: Predice la probabilidad de ganar desde la posicion actual (-1 a +1)

Arquitectura:
- Input: Tablero de Connect4 como tensor de 3 canales (6x7)
- Cuerpo: Torre de bloques residuales (ResNet)
- Output: Policy logits (7) + Value (-1 a +1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class ResidualBlock(nn.Module):
    """
    Bloque Residual (Skip Connection) para redes profundas.
    
    La idea clave de ResNet es que en lugar de aprender F(x) directamente,
    aprendemos F(x) + x. Esto permite que los gradientes fluyan directamente
    a traves de la red, evitando el problema del "vanishing gradient".
    
    Arquitectura:
        input -> Conv -> BN -> ReLU -> Conv -> BN -> (+input) -> ReLU -> output
                  |___________________________________|
                            Skip Connection
    
    Args:
        num_filters: Numero de filtros en las capas convolucionales.
    """
    
    def __init__(self, num_filters: int) -> None:
        """
        Inicializa el bloque residual.
        
        Args:
            num_filters: Numero de filtros para las convoluciones.
        """
        super().__init__()
        
        # Primera capa convolucional
        # kernel_size=3 con padding=1 mantiene las dimensiones espaciales
        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False  # No necesitamos bias porque BatchNorm lo incluye
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del bloque residual.
        
        Implementa: output = ReLU(F(x) + x)
        donde F(x) = BN(Conv(ReLU(BN(Conv(x)))))
        
        Args:
            x: Tensor de entrada de forma (batch, filters, height, width)
            
        Returns:
            Tensor de salida con la misma forma que la entrada.
        """
        # Guardar la entrada para la skip connection
        identity = x
        
        # Primera transformacion: Conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Segunda transformacion: Conv -> BN (sin ReLU todavia)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection: sumar la entrada original
        # Esta es la clave de ResNet - permite que los gradientes fluyan directamente
        out = out + identity
        
        # ReLU final despues de la suma
        out = F.relu(out)
        
        return out


class Connect4Net(nn.Module):
    """
    Red neuronal principal para AlphaZero-Mini (Connect4).
    
    Esta red toma un tablero de Connect4 como entrada y produce:
    1. Policy: Probabilidades (logits) para cada columna (7 valores)
    2. Value: Evaluacion de la posicion (-1 = derrota, +1 = victoria)
    
    Arquitectura:
        Input (3, 6, 7)
            |
        [Bloque Inicial: Conv -> BN -> ReLU]
            |
        [Torre Residual: N bloques ResidualBlock]
            |
        +---+---+
        |       |
    [Policy]  [Value]
        |       |
    (7 logits) (1 valor)
    
    Args:
        num_res_blocks: Numero de bloques residuales en la torre (default: 4)
        num_filters: Numero de filtros en las capas convolucionales (default: 64)
    """
    
    # Dimensiones del tablero de Connect4
    BOARD_HEIGHT: int = 6
    BOARD_WIDTH: int = 7
    INPUT_CHANNELS: int = 3  # Piezas propias, piezas oponente, celdas vacias
    NUM_ACTIONS: int = 7     # Una accion por columna
    
    def __init__(
        self,
        num_res_blocks: int = 4,
        num_filters: int = 64
    ) -> None:
        """
        Inicializa la red Connect4Net.
        
        Args:
            num_res_blocks: Numero de bloques residuales (default: 4)
            num_filters: Numero de filtros por capa conv (default: 64)
        """
        super().__init__()
        
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        
        # =====================================================================
        # BLOQUE INICIAL DE CONVOLUCION
        # Transforma el input de 3 canales a num_filters canales
        # =====================================================================
        self.initial_conv = nn.Conv2d(
            in_channels=self.INPUT_CHANNELS,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.initial_bn = nn.BatchNorm2d(num_filters)
        
        # =====================================================================
        # TORRE RESIDUAL
        # Stack de bloques residuales que extraen caracteristicas
        # =====================================================================
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # =====================================================================
        # POLICY HEAD (Cabeza de Politica)
        # Predice la probabilidad de jugar en cada columna
        # =====================================================================
        self.policy_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=32,
            kernel_size=1,  # Convolucion 1x1 para reducir canales
            bias=False
        )
        self.policy_bn = nn.BatchNorm2d(32)
        # Tamanio despues de flatten: 32 canales * 6 filas * 7 columnas = 1344
        self.policy_fc = nn.Linear(32 * self.BOARD_HEIGHT * self.BOARD_WIDTH, self.NUM_ACTIONS)
        
        # =====================================================================
        # VALUE HEAD (Cabeza de Valor)
        # Predice el valor de la posicion (-1 a +1)
        # =====================================================================
        self.value_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=3,
            kernel_size=1,  # Convolucion 1x1 para reducir canales
            bias=False
        )
        self.value_bn = nn.BatchNorm2d(3)
        # Tamanio despues de flatten: 3 canales * 6 filas * 7 columnas = 126
        self.value_fc1 = nn.Linear(3 * self.BOARD_HEIGHT * self.BOARD_WIDTH, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass de la red.
        
        Args:
            x: Tensor de entrada de forma (batch_size, 3, 6, 7)
               - Canal 0: Piezas del jugador actual
               - Canal 1: Piezas del oponente
               - Canal 2: Celdas vacias
        
        Returns:
            Tupla (policy_logits, value):
                - policy_logits: Tensor (batch_size, 7) - logits sin softmax
                - value: Tensor (batch_size, 1) - valor entre -1 y +1
        """
        # =====================================================================
        # BLOQUE INICIAL
        # =====================================================================
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)
        
        # =====================================================================
        # TORRE RESIDUAL
        # =====================================================================
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # =====================================================================
        # POLICY HEAD
        # =====================================================================
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        # Flatten: (batch, 32, 6, 7) -> (batch, 32*6*7)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        # NO aplicamos softmax aqui - se aplica durante entrenamiento/inferencia
        
        # =====================================================================
        # VALUE HEAD
        # =====================================================================
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        # Flatten: (batch, 3, 6, 7) -> (batch, 3*6*7)
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        # Tanh para obtener valor entre -1 y +1
        value = torch.tanh(value)
        
        return policy_logits, value
    
    def predict(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Realiza prediccion con la red (modo evaluacion).
        
        A diferencia de forward(), este metodo:
        - Aplica softmax a los policy logits
        - Aplica temperatura para controlar exploracion
        - Devuelve numpy arrays en lugar de tensors
        
        Args:
            x: Tensor de entrada (1, 3, 6, 7) - un solo tablero
            temperature: Controla exploracion (1.0=normal, <1=determinista, >1=aleatorio)
        
        Returns:
            Tupla (policy_probs, value):
                - policy_probs: Array numpy (7,) con probabilidades
                - value: Float entre -1 y +1
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            
            # Aplicar temperatura a los logits antes del softmax
            if temperature != 1.0:
                policy_logits = policy_logits / temperature
            
            # Softmax para obtener probabilidades
            policy_probs = F.softmax(policy_logits, dim=1)
            
            return policy_probs.cpu().numpy()[0], value.cpu().item()


def board_to_tensor(board: np.ndarray, player: int) -> torch.Tensor:
    """
    Convierte un tablero de Connect4 a un tensor de 3 canales.
    
    El tablero se transforma a la perspectiva del jugador actual
    (canonical board) y se codifica en 3 canales binarios:
    
    - Canal 0: Piezas del jugador actual (1 donde hay pieza, 0 si no)
    - Canal 1: Piezas del oponente (1 donde hay pieza, 0 si no)  
    - Canal 2: Celdas vacias (1 donde esta vacio, 0 si no)
    
    Args:
        board: Tablero numpy de forma (6, 7) con valores {-1, 0, 1}
        player: Jugador actual (1 o -1)
    
    Returns:
        Tensor de forma (3, 6, 7) como float32
    
    Example:
        >>> board = np.zeros((6, 7), dtype=np.int8)
        >>> board[5, 3] = 1  # Jugador 1 en columna 3
        >>> tensor = board_to_tensor(board, player=1)
        >>> tensor.shape
        torch.Size([3, 6, 7])
    """
    # Transformar a perspectiva del jugador actual (canonical board)
    # Si player=1: tablero queda igual
    # Si player=-1: signos invertidos (las piezas del jugador pasan a ser 1)
    canonical_board = board * player
    
    # Crear los 3 canales
    # Canal 0: Piezas del jugador actual (valor 1 en canonical)
    current_player_pieces = (canonical_board == 1).astype(np.float32)
    
    # Canal 1: Piezas del oponente (valor -1 en canonical)
    opponent_pieces = (canonical_board == -1).astype(np.float32)
    
    # Canal 2: Celdas vacias (valor 0)
    empty_cells = (canonical_board == 0).astype(np.float32)
    
    # Apilar los canales: (3, 6, 7)
    tensor = np.stack([current_player_pieces, opponent_pieces, empty_cells], axis=0)
    
    return torch.from_numpy(tensor)


def save_checkpoint(
    model: Connect4Net,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    filepath: str
) -> None:
    """
    Guarda un checkpoint del modelo y optimizador.
    
    El checkpoint incluye:
    - Pesos del modelo (state_dict)
    - Estado del optimizador
    - Numero de iteracion actual
    - Configuracion de la red (num_res_blocks, num_filters)
    
    Args:
        model: Instancia de Connect4Net
        optimizer: Optimizador de PyTorch
        iteration: Numero de iteracion de entrenamiento
        filepath: Ruta donde guardar el checkpoint
    
    Example:
        >>> model = Connect4Net()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> save_checkpoint(model, optimizer, iteration=10, filepath='models/checkpoint_10.pt')
    """
    # Crear directorio padre si no existe
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'num_res_blocks': model.num_res_blocks,
        'num_filters': model.num_filters,
    }
    
    torch.save(checkpoint, filepath)
    print(f"[Checkpoint] Guardado en: {filepath}")


def load_checkpoint(
    filepath: str,
    model: Connect4Net,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """
    Carga un checkpoint y restaura el modelo y optimizador.
    
    Args:
        filepath: Ruta del archivo de checkpoint
        model: Instancia de Connect4Net donde cargar los pesos
        optimizer: Optimizador para restaurar (opcional)
    
    Returns:
        Numero de iteracion del checkpoint
    
    Raises:
        FileNotFoundError: Si el archivo no existe
    
    Example:
        >>> model = Connect4Net()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> iteration = load_checkpoint('models/checkpoint_10.pt', model, optimizer)
        >>> print(f"Continuando desde iteracion {iteration}")
    """
    checkpoint = torch.load(filepath, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    iteration = checkpoint.get('iteration', 0)
    print(f"[Checkpoint] Cargado desde: {filepath} (iteracion {iteration})")
    
    return iteration


def count_parameters(model: nn.Module) -> int:
    """
    Cuenta el numero total de parametros entrenables en un modelo.
    
    Args:
        model: Cualquier modelo de PyTorch
    
    Returns:
        Numero total de parametros entrenables
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# TESTS - Ejecutar con: python neural_net.py
# =============================================================================
if __name__ == "__main__":
    import os
    import tempfile
    
    print("=" * 60)
    print("TESTS DE NEURAL NETWORK")
    print("=" * 60)
    
    # Detectar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Dispositivo: {device}")
    
    # -------------------------------------------------------------------------
    # TEST 1: Creacion de la red
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Creacion de la red Connect4Net")
    
    model = Connect4Net(num_res_blocks=4, num_filters=64)
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"[OK] Red creada exitosamente")
    print(f"     - Bloques residuales: {model.num_res_blocks}")
    print(f"     - Filtros por capa: {model.num_filters}")
    print(f"     - Parametros entrenables: {num_params:,}")
    
    # -------------------------------------------------------------------------
    # TEST 2: Conversion de tablero a tensor
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Conversion de tablero a tensor")
    
    # Crear un tablero de prueba
    test_board = np.zeros((6, 7), dtype=np.int8)
    test_board[5, 3] = 1   # Jugador 1 en columna 3
    test_board[5, 4] = -1  # Jugador -1 en columna 4
    test_board[4, 3] = 1   # Jugador 1 encima
    
    print("Tablero de prueba:")
    print(test_board)
    
    # Convertir desde perspectiva del jugador 1
    tensor_p1 = board_to_tensor(test_board, player=1)
    print(f"\n[OK] Tensor creado (perspectiva jugador 1)")
    print(f"     Forma: {tensor_p1.shape}")
    
    # Verificar forma
    assert tensor_p1.shape == (3, 6, 7), f"Forma incorrecta: {tensor_p1.shape}"
    print("     [OK] Forma correcta: (3, 6, 7)")
    
    # Verificar que los canales suman 1 en cada posicion
    channel_sum = tensor_p1.sum(dim=0)
    assert torch.all(channel_sum == 1), "Los canales no suman 1"
    print("     [OK] Canales son mutuamente excluyentes")
    
    # Convertir desde perspectiva del jugador -1
    tensor_p2 = board_to_tensor(test_board, player=-1)
    print(f"\n[OK] Tensor creado (perspectiva jugador -1)")
    # Los canales 0 y 1 deben estar invertidos
    assert torch.equal(tensor_p1[0], tensor_p2[1]), "Canales no invertidos correctamente"
    assert torch.equal(tensor_p1[1], tensor_p2[0]), "Canales no invertidos correctamente"
    print("     [OK] Perspectiva canonica funciona correctamente")
    
    # -------------------------------------------------------------------------
    # TEST 3: Forward pass
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Forward pass de la red")
    
    # Crear batch de entrada
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 6, 7).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        policy_logits, value = model(dummy_input)
    
    print(f"[OK] Forward pass exitoso")
    print(f"     - Input shape: {dummy_input.shape}")
    print(f"     - Policy logits shape: {policy_logits.shape}")
    print(f"     - Value shape: {value.shape}")
    
    # Verificar formas de salida
    assert policy_logits.shape == (batch_size, 7), f"Policy shape incorrecta: {policy_logits.shape}"
    assert value.shape == (batch_size, 1), f"Value shape incorrecta: {value.shape}"
    print("     [OK] Formas de salida correctas")
    
    # Verificar que value esta en rango [-1, 1]
    assert torch.all(value >= -1) and torch.all(value <= 1), "Value fuera de rango"
    print("     [OK] Value en rango [-1, 1]")
    
    # -------------------------------------------------------------------------
    # TEST 4: Metodo predict
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Metodo predict")
    
    # Usar el tablero de prueba
    single_input = tensor_p1.unsqueeze(0).to(device)  # Agregar dimension batch
    
    policy_probs, value_scalar = model.predict(single_input)
    
    print(f"[OK] Prediccion exitosa")
    print(f"     - Policy probs: {policy_probs}")
    print(f"     - Value: {value_scalar:.4f}")
    
    # Verificar que policy_probs suma 1
    prob_sum = policy_probs.sum()
    assert abs(prob_sum - 1.0) < 1e-5, f"Probabilidades no suman 1: {prob_sum}"
    print(f"     [OK] Probabilidades suman: {prob_sum:.6f}")
    
    # -------------------------------------------------------------------------
    # TEST 5: Save/Load checkpoint
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Save/Load checkpoint")
    
    # Crear optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Guardar checkpoint en directorio temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
        
        # Guardar
        save_checkpoint(model, optimizer, iteration=42, filepath=checkpoint_path)
        
        # Crear nuevo modelo y cargar
        new_model = Connect4Net(num_res_blocks=4, num_filters=64).to(device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        loaded_iteration = load_checkpoint(checkpoint_path, new_model, new_optimizer)
        
        assert loaded_iteration == 42, f"Iteracion incorrecta: {loaded_iteration}"
        print(f"     [OK] Iteracion cargada: {loaded_iteration}")
        
        # Verificar que los pesos son iguales
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), 
            new_model.named_parameters()
        ):
            assert torch.equal(param1, param2), f"Pesos diferentes en {name1}"
        print("     [OK] Pesos del modelo restaurados correctamente")
    
    # -------------------------------------------------------------------------
    # TEST 6: Diferentes configuraciones
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Diferentes configuraciones de red")
    
    configs = [
        (2, 32),   # Pequena
        (4, 64),   # Default
        (6, 128),  # Grande
    ]
    
    for num_blocks, num_filters in configs:
        test_model = Connect4Net(num_res_blocks=num_blocks, num_filters=num_filters)
        params = count_parameters(test_model)
        print(f"     - {num_blocks} bloques, {num_filters} filtros: {params:,} parametros")
    
    print("     [OK] Todas las configuraciones funcionan")
    
    # -------------------------------------------------------------------------
    # TEST 7: GPU/CPU device placement
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Device placement")
    
    cpu_model = Connect4Net().to('cpu')
    cpu_input = torch.randn(1, 3, 6, 7)
    cpu_policy, cpu_value = cpu_model(cpu_input)
    print(f"     [OK] CPU: policy device={cpu_policy.device}, value device={cpu_value.device}")
    
    if torch.cuda.is_available():
        gpu_model = Connect4Net().to('cuda')
        gpu_input = torch.randn(1, 3, 6, 7).to('cuda')
        gpu_policy, gpu_value = gpu_model(gpu_input)
        print(f"     [OK] GPU: policy device={gpu_policy.device}, value device={gpu_value.device}")
    else:
        print("     [SKIP] CUDA no disponible")
    
    # -------------------------------------------------------------------------
    # Resumen final
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[OK] TODOS LOS TESTS PASARON CORRECTAMENTE")
    print("=" * 60)
    print(f"\nResumen de la red Connect4Net:")
    print(f"  - Input:  (batch, 3, 6, 7) - Tablero con 3 canales")
    print(f"  - Output: Policy (batch, 7) + Value (batch, 1)")
    print(f"  - Parametros: {count_parameters(model):,}")
    print("=" * 60)

