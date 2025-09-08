class WSiLU(nn.Module):
    # ===== Class-level (static) attributes =====
    _noise_enabled: bool = False
    _noise_amp: float = 1e-8
    _noise_intervals: List[Tuple[float, float]] = []  # closed intervals [low, high]

    # Opcional: controlar promoção para fp32 em atv. (bom para fp16)
    _promote_activations: bool = True

    def __init__(self):
        super().__init__()

    # ===== Static methods (enable/disable) =====
    @staticmethod
    def enable_noise() -> None:
        WSiLU._noise_enabled = True

    @staticmethod
    def disable_noise() -> None:
        WSiLU._noise_enabled = False

    @staticmethod
    def is_noise_enabled() -> bool:
        return WSiLU._noise_enabled

    # ===== Static methods (getters/setters) =====
    @staticmethod
    def set_noise_amp(amp: float) -> None:
        if amp < 0:
            raise ValueError("noise_amp must be >= 0.")
        WSiLU._noise_amp = float(amp)

    @staticmethod
    def get_noise_amp() -> float:
        return WSiLU._noise_amp

    @staticmethod
    def set_noise_intervals(intervals: List[Tuple[float, float]]) -> None:
        checked = []
        for low, high in intervals:
            low = float(low); high = float(high)
            if high < low:
                raise ValueError(f"Invalid interval: ({low}, {high})")
            checked.append((low, high))
        WSiLU._noise_intervals = checked

    @staticmethod
    def add_noise_interval(low: float, high: float) -> None:
        if high < low:
            raise ValueError(f"Invalid interval: ({low}, {high})")
        WSiLU._noise_intervals.append((float(low), float(high)))

    @staticmethod
    def clear_noise_intervals() -> None:
        WSiLU._noise_intervals = []

    @staticmethod
    def get_noise_intervals() -> List[Tuple[float, float]]:
        return list(WSiLU._noise_intervals)

    @staticmethod
    def set_promote_activations(flag: bool) -> None:
        """Habilita/desabilita promoção para float32 dentro do forward (recomendado p/ fp16)."""
        WSiLU._promote_activations = bool(flag)

    @staticmethod
    def get_promote_activations() -> bool:
        return WSiLU._promote_activations

    # ===== Forward pass =====
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Escolhe dtype de trabalho para a ativação: promove para fp32 quando for fp16/bf16, se habilitado
        compute_in_fp32 = (x.dtype in (torch.float16, torch.bfloat16)) and WSiLU._promote_activations

        if compute_in_fp32:
            x_work = x.to(torch.float32)
            four = torch.tensor(4.0, dtype=torch.float32, device=x.device)
            y = torch.sigmoid(four * x_work) * x_work
            y = y.to(x.dtype)  # volta ao dtype original
        else:
            # Mantém dtype original, mas garante literal no mesmo dtype/device
            four = torch.tensor(4.0, dtype=x.dtype, device=x.device)
            y = torch.sigmoid(four * x) * x

        # Sem ruído?
        if (not WSiLU._noise_enabled) or (WSiLU._noise_amp == 0.0) or (len(WSiLU._noise_intervals) == 0):
            return y

        # Máscara booleana: True onde x cai em algum intervalo
        # Compare em dtype do próprio x (evita casts implícitos)
        mask = torch.zeros_like(x, dtype=torch.bool)
        for low, high in WSiLU._noise_intervals:
            low_t = torch.tensor(low, dtype=x.dtype, device=x.device)
            high_t = torch.tensor(high, dtype=x.dtype, device=x.device)
            mask |= (x >= low_t) & (x <= high_t)

        if mask.any():
            # Define amplitude no dtype/device corretos
            amp = torch.tensor(WSiLU._noise_amp, dtype=y.dtype, device=y.device)

            # Epsilon para garantir que não cruzaremos o zero (estritamente mesmo sinal)
            eps = torch.tensor(torch.finfo(y.dtype).eps, dtype=y.dtype, device=y.device) * 4

            # Ruído base ~ U(-amp, amp)
            noise = torch.zeros_like(y)
            base = torch.empty_like(y[mask]).uniform_(-1.0, 1.0) * amp

            y_m = y[mask]

            # Limites por elemento para não mudar o sinal de y
            # Caso y > 0: noise >= -y + eps   (upper = +amp)
            # Caso y < 0: noise <= -y - eps   (lower = -amp)
            # Caso y == 0: noise = 0
            lower = torch.full_like(y_m, -amp)
            upper = torch.full_like(y_m,  amp)

            pos = y_m > 0
            neg = y_m < 0
            zer = ~pos & ~neg

            # Ajusta limites para manter o sinal
            lower[pos] = torch.maximum(lower[pos], -y_m[pos] + eps)  # não pode ir além de -y + eps
            upper[neg] = torch.minimum(upper[neg], -y_m[neg] - eps)  # não pode ir além de -y - eps

            # Onde y == 0, força ruído zero
            lower[zer] = 0
            upper[zer] = 0

            # Caso degenerado: se por precisão lower > upper, zera ruído
            invalid = lower > upper

            # Aplica os limites ao ruído base
            clipped = torch.maximum(torch.minimum(base, upper), lower)
            clipped[invalid] = 0.0

            noise[mask] = clipped
            y = y + noise

        return y

    
    # def forward_polinomial(self, x): #GRAU 6
    # def forward(self, x):
    #     # parte polinomial
    #     poly = (
    #         0.004546
    #         + 0.5 * x
    #         + 0.859845 * x**2
    #         - 0.553798 * x**4
    #         + 0.168839 * x**6
    #     )

    #     # aplica as condições em cada ponto
    #     return torch.where(
    #         x < -1.2,
    #         torch.zeros_like(x),  # se x < -1.2
    #         torch.where(x > 1.2, x, poly)  # se x > 1.2 → x, senão → poly
    #     )

    # def wsilu_polinomial_forward(x: torch.Tensor) -> torch.Tensor: #GRAU 16
    # def forward(self, x) :
    #     x2 = x * x
    #     poly_even = (
    #         (((((((( -0.00047333 * x2 + 0.0083775) * x2 - 0.06236471) * x2
    #                 + 0.25503359) * x2 - 0.63088907) * x2 + 0.99181597) * x2
    #             - 1.04954556) * x2 + 0.96917012) * x2 + 0.00059508
    #         )
    #     )
    #     poly = poly_even + 0.5 * x
    #     return torch.where(x < -2, torch.zeros_like(x),
    #         torch.where(x > 2, x, poly))
        
    
import torch
import torch.nn as nn

class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # aloca a saída
        y = torch.empty_like(x)

        # x < -2  -> 0
        m = (x < -2)
        y[m] = 0.0

        # -2 <= x < -1.5
        m = (x >= -2) & (x < -1.5)
        xm = x[m]
        y[m] = -0.04077822 - 0.03895519*xm - 0.00946458*xm**2

        # -1.5 <= x < -1
        m = (x >= -1.5) & (x < -1)
        xm = x[m]
        y[m] = -0.10496491 - 0.12680174*xm - 0.03961572*xm**2

        # -1 <= x < -0.5
        m = (x >= -1) & (x < -0.5)
        xm = x[m]
        y[m] = -0.12970709 - 0.16552587*xm - 0.05347109*xm**2

        # -0.5 <= x < -0.25
        m = (x >= -0.5) & (x < -0.25)
        xm = x[m]
        y[m] = -0.03668002 + 0.20407026*xm + 0.31848767*xm**2

        # -0.25 <= x < 0
        m = (x >= -0.25) & (x < 0)
        xm = x[m]
        y[m] = -0.00038846 + 0.48311933*xm + 0.87066946*xm**2

        # 0 <= x < 0.25
        m = (x >= 0) & (x < 0.25)
        xm = x[m]
        y[m] = -0.00038846 + 0.51688067*xm + 0.87066946*xm**2

        # 0.25 <= x < 0.5
        m = (x >= 0.25) & (x < 0.5)
        xm = x[m]
        y[m] = -0.03668002 + 0.79592974*xm + 0.31848767*xm**2

        # 0.5 <= x < 1
        m = (x >= 0.5) & (x < 1)
        xm = x[m]
        y[m] = -0.12970709 + 1.16552587*xm - 0.05347109*xm**2

        # 1 <= x < 1.5
        m = (x >= 1) & (x < 1.5)
        xm = x[m]
        y[m] = -0.10496491 + 1.12680174*xm - 0.03961572*xm**2

        # 1.5 <= x < 2
        m = (x >= 1.5) & (x < 2)
        xm = x[m]
        y[m] = -0.04077822 + 1.03895519*xm - 0.00946458*xm**2

        # x >= 2  -> identidade
        m = (x >= 2)
        y[m] = x[m]

        return y
