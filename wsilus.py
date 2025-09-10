
#ORIGINAL
class WSiLU(nn.Module):
    def forward(self, x):
        y = torch.sigmoid(4.0 * x) * x
        return y


#NOISEY
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


class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_normal(self, x):
        y = torch.sigmoid(4.0 * x) * x
        return y
    
    # def forward_polinomial(self, x): #GRAU 6
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


#VANESSA LUT ASSIMETRICO 4 INTERVALOS 512 (lut_asyn_4int_512entries)
class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- LUT em float16 (mesma lista hex do seu código) -----
        lut_hex = [
            0x3800, 0x3804, 0x3808, 0x380c, 0x3810, 0x3814, 0x3818, 0x381c, 0x3820, 0x3824, 0x3828, 0x382c, 0x3830, 0x3834, 0x3838, 0x383c, 0x3840, 0x3844, 0x3848, 0x384c, 0x3850, 0x3854, 0x3858, 0x385c, 0x3860, 0x3864, 0x3868, 0x386c, 0x3870, 0x3874, 0x3877, 0x387b, 0x387f, 0x3883, 0x3887, 0x388b, 0x388f, 0x3893, 0x3897, 0x389b, 0x389f, 0x38a3, 0x38a7, 0x38aa, 0x38ae, 0x38b2, 0x38b6, 0x38ba, 0x38be, 0x38c2, 0x38c5, 0x38c9, 0x38cd, 0x38d1, 0x38d5, 0x38d9, 0x38dc, 0x38e0, 0x38e4, 0x38e8, 0x38ec, 0x38ef, 0x38f3, 0x38f7, 0x38fb, 0x38ff, 0x3902, 0x3906, 0x390a, 0x390e, 0x3911, 0x3915, 0x3919, 0x391c, 0x3920, 0x3924, 0x3927, 0x392b, 0x392f, 0x3932, 0x3936, 0x393a, 0x393d, 0x3941, 0x3944, 0x3948, 0x394c, 0x394f, 0x3953, 0x3956, 0x395a, 0x395d, 0x3961, 0x3964, 0x3968, 0x396b, 0x396f, 0x3972, 0x3976, 0x3979, 0x397d, 0x3980, 0x3984, 0x3987, 0x398b, 0x398e, 0x3991, 0x3995, 0x3998, 0x399b, 0x399f, 0x39a2, 0x39a5, 0x39a9, 0x39ac, 0x39af, 0x39b3, 0x39b6, 0x39b9, 0x39bc, 0x39c0, 0x39c3, 0x39c6, 0x39c9, 0x39cd, 0x39d0, 0x39d3, 0x39d6, 0x39d9, 0x39dc, 0x39df, 0x39e3, 0x39e6, 0x39e9, 0x39ec, 0x39ef, 0x39f2, 0x39f5, 0x39f8, 0x39fb, 0x39fe, 0x3a01, 0x3a04, 0x3a07, 0x3a0a, 0x3a0d, 0x3a10, 0x3a13, 0x3a16, 0x3a19, 0x3a1c, 0x3a1e, 0x3a21, 0x3a24, 0x3a27, 0x3a2a, 0x3a2d, 0x3a30, 0x3a32, 0x3a35, 0x3a38, 0x3a3b, 0x3a3d, 0x3a40, 0x3a43, 0x3a46, 0x3a48, 0x3a4b, 0x3a4e, 0x3a50, 0x3a53, 0x3a56, 0x3a58, 0x3a5b, 0x3a5e, 0x3a60, 0x3a63, 0x3a65, 0x3a68, 0x3a6a, 0x3a6d, 0x3a6f, 0x3a72, 0x3a74, 0x3a77, 0x3a79, 0x3a7c, 0x3a7e, 0x3a81, 0x3a83, 0x3a86, 0x3a88, 0x3a8a, 0x3a8d, 0x3a8f, 0x3a91, 0x3a94, 0x3a96, 0x3a98, 0x3a9b, 0x3a9d, 0x3a9f, 0x3aa2, 0x3aa4, 0x3aa6, 0x3aa8, 0x3aab, 0x3aad, 0x3aaf, 0x3ab1, 0x3ab3, 0x3ab6, 0x3ab8, 0x3aba, 0x3abc, 0x3abe, 0x3ac0, 0x3ac2, 0x3ac4, 0x3ac7, 0x3ac9, 0x3acb, 0x3acd, 0x3acf, 0x3ad1, 0x3ad3, 0x3ad5, 0x3ad7, 0x3ad9, 0x3adb, 0x3add, 0x3adf, 0x3ae1, 0x3ae3, 0x3ae4, 0x3ae6, 0x3ae8, 0x3aea, 0x3aec, 0x3aee, 0x3af0, 0x3af2, 0x3af3, 0x3af5, 0x3af7, 0x3af9, 0x3afb, 0x3afc, 0x3afe, 0x3b00, 0x3b02, 0x3b03, 0x3b05, 0x3b07, 0x3b08, 0x3b0a, 0x3b0c, 0x3b0f, 0x3b13, 0x3b16, 0x3b19, 0x3b1c, 0x3b1f, 0x3b22, 0x3b25, 0x3b29, 0x3b2c, 0x3b2e, 0x3b31, 0x3b34, 0x3b37, 0x3b3a, 0x3b3d, 0x3b3f, 0x3b42, 0x3b45, 0x3b47, 0x3b4a, 0x3b4d, 0x3b4f, 0x3b52, 0x3b54, 0x3b57, 0x3b59, 0x3b5b, 0x3b5e, 0x3b60, 0x3b62, 0x3b65, 0x3b67, 0x3b69, 0x3b6b, 0x3b6d, 0x3b6f, 0x3b72, 0x3b74, 0x3b76, 0x3b78, 0x3b7a, 0x3b7c, 0x3b7e, 0x3b7f, 0x3b81, 0x3b83, 0x3b85, 0x3b87, 0x3b89, 0x3b8a, 0x3b8c, 0x3b8e, 0x3b8f, 0x3b91, 0x3b93, 0x3b94, 0x3b96, 0x3b97, 0x3b99, 0x3b9a, 0x3b9c, 0x3b9d, 0x3b9f, 0x3ba0, 0x3ba2, 0x3ba3, 0x3ba4, 0x3ba6, 0x3ba7, 0x3ba9, 0x3baa, 0x3bab, 0x3bac, 0x3bae, 0x3baf, 0x3bb0, 0x3bb1, 0x3bb2, 0x3bb4, 0x3bb5, 0x3bb6, 0x3bb7, 0x3bb8, 0x3bb9, 0x3bba, 0x3bbb, 0x3bbc, 0x3bbd, 0x3bbe, 0x3bbf, 0x3bc0, 0x3bc1, 0x3bc2, 0x3bc3, 0x3bc4, 0x3bc5, 0x3bc6, 0x3bc7, 0x3bc8, 0x3bc8, 0x3bc9, 0x3bca, 0x3bcb, 0x3bcc, 0x3bcc, 0x3bcd, 0x3bce, 0x3bcf, 0x3bcf, 0x3bd0, 0x3bd1, 0x3bd2, 0x3bd2, 0x3bd3, 0x3bd4, 0x3bd4, 0x3bd5, 0x3bd6, 0x3bd6, 0x3bd7, 0x3bd8, 0x3bd8, 0x3bd9, 0x3bd9, 0x3bda, 0x3bdb, 0x3bdb, 0x3bdc, 0x3bdd, 0x3bde, 0x3bdf, 0x3be0, 0x3be1, 0x3be2, 0x3be3, 0x3be4, 0x3be5, 0x3be6, 0x3be7, 0x3be7, 0x3be8, 0x3be9, 0x3be9, 0x3bea, 0x3beb, 0x3beb, 0x3bec, 0x3bed, 0x3bed, 0x3bee, 0x3bee, 0x3bef, 0x3bef, 0x3bf0, 0x3bf0, 0x3bf1, 0x3bf1, 0x3bf2, 0x3bf2, 0x3bf3, 0x3bf3, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf5, 0x3bf5, 0x3bf5, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3c00,
        ]

        lut_float16 = np.array(
            [struct.unpack('>e', struct.pack('>H', v))[0] for v in lut_hex],
            dtype=np.float16
        )
        # registra como buffer para mover com .to(device) e salvar no state_dict
        self.register_buffer("lut", torch.from_numpy(lut_float16))  # float16

        # Constantes (iguais ao seu VHDL)
        self.LIMIT_1 = 0.5
        self.LIMIT_2 = 1.0
        self.LIMIT_3 = 1.5
        self.LIMIT_4 = 2.0
        self.N_1 = 256
        self.N_2 = 128
        self.N_3 = 64
        self.N_4 = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype

        # LUT no mesmo device/dtype do input
        lut = self.lut.to(device=device, dtype=dtype)              # (512,)
        lut_size = lut.numel()

        xq = x  # já está no dtype desejado
        absx = xq.abs()

        # índices inteiros (0 .. lut_size-1)
        idx = torch.empty_like(xq, dtype=torch.long)

        # máscaras por intervalo (mesma lógica do seu código)
        m1 = absx <= self.LIMIT_1
        m2 = (absx > self.LIMIT_1) & (absx <= self.LIMIT_2)
        m3 = (absx > self.LIMIT_2) & (absx <= self.LIMIT_3)
        m4 = (absx > self.LIMIT_3) & (absx <= self.LIMIT_4)
        m5 = absx > self.LIMIT_4

        # int() em Python trunca para baixo para números positivos -> use floor
        if m1.any():
            idx[m1] = torch.floor((absx[m1] / self.LIMIT_1) * self.N_1).to(torch.long)

        if m2.any():
            idx[m2] = ( self.N_1
                        + torch.floor(((absx[m2] - self.LIMIT_1) / (self.LIMIT_2 - self.LIMIT_1)) * self.N_2).to(torch.long) )

        if m3.any():
            idx[m3] = ( self.N_1 + self.N_2
                        + torch.floor(((absx[m3] - self.LIMIT_2) / (self.LIMIT_3 - self.LIMIT_2)) * self.N_3).to(torch.long) )

        if m4.any():
            idx[m4] = ( self.N_1 + self.N_2 + self.N_3
                        + torch.floor(((absx[m4] - self.LIMIT_3) / (self.LIMIT_4 - self.LIMIT_3)) * self.N_4).to(torch.long) )

        if m5.any():
            idx[m5] = lut_size - 1

        # garante 0..lut_size-1 (cobre bordas como 0.5, 1.0, 1.5, 2.0)
        idx = idx.clamp_(min=0, max=lut_size - 1)

        # valor de sigmoid a partir da LUT
        sig_val = lut[idx]

        # simetria para x < 0 (e zerar para x < -2)
        one = torch.tensor(1.0, dtype=dtype, device=device)
        y_sigmoid = torch.where(xq < 0, one - sig_val, sig_val)
        y_sigmoid = torch.where(xq < -self.LIMIT_4, torch.zeros_like(y_sigmoid), y_sigmoid)

        return xq * y_sigmoid
    

#LUT SYMETRIC
# class WSiLU(torch.nn.Module):
#     """
#     wsilu_lut_un: usa LUT uniforme em |x|<=2.0 e clampa fora.
#     Saída = x * sigmoid_aprox(|x|), com simetria para x<0.
#     A LUT é registrada como buffer não-persistente (não entra no state_dict).
#     """
#     def __init__(self, device=None, dtype=torch.float16):
#         super().__init__()

#         # ---- monta LUT (float16) a partir dos hex fornecidos ----
#         lut_hex = [
#             0x3800, 0x380c, 0x3818, 0x3824, 0x3830, 0x383c, 0x3848, 0x3854, 0x3860, 0x386c, 0x3878, 0x3884, 0x388f, 0x389b, 0x38a7,
#             0x38b3, 0x38be, 0x38ca, 0x38d5, 0x38e1, 0x38ec, 0x38f7, 0x3903, 0x390e, 0x3919, 0x3924, 0x392f, 0x393a, 0x3945, 0x3950,
#             0x395a, 0x3965, 0x3970, 0x397a, 0x3984, 0x398f, 0x3999, 0x39a3, 0x39ad, 0x39b7, 0x39c0, 0x39ca, 0x39d4, 0x39dd, 0x39e7,
#             0x39f0, 0x39f9, 0x3a02, 0x3a0b, 0x3a14, 0x3a1c, 0x3a25, 0x3a2e, 0x3a36, 0x3a3e, 0x3a46, 0x3a4f, 0x3a57, 0x3a5e, 0x3a66,
#             0x3a6e, 0x3a75, 0x3a7d, 0x3a84, 0x3a8b, 0x3a92, 0x3a99, 0x3aa0, 0x3aa7, 0x3aae, 0x3ab4, 0x3abb, 0x3ac1, 0x3ac7, 0x3ace,
#             0x3ad4, 0x3ada, 0x3ae0, 0x3ae5, 0x3aeb, 0x3af1, 0x3af6, 0x3afb, 0x3b01, 0x3b06, 0x3b0b, 0x3b10, 0x3b15, 0x3b1a, 0x3b1f,
#             0x3b23, 0x3b28, 0x3b2c, 0x3b31, 0x3b35, 0x3b39, 0x3b3e, 0x3b42, 0x3b46, 0x3b4a, 0x3b4d, 0x3b51, 0x3b55, 0x3b59, 0x3b5c,
#             0x3b60, 0x3b63, 0x3b66, 0x3b6a, 0x3b6d, 0x3b70, 0x3b73, 0x3b76, 0x3b79, 0x3b7c, 0x3b7f, 0x3b82, 0x3b85, 0x3b87, 0x3b8a,
#             0x3b8d, 0x3b8f, 0x3b92, 0x3b94, 0x3b96, 0x3b99, 0x3b9b, 0x3b9d, 0x3b9f, 0x3ba2, 0x3ba4, 0x3ba6, 0x3ba8, 0x3baa, 0x3bac,
#             0x3bad, 0x3baf, 0x3bb1, 0x3bb3, 0x3bb5, 0x3bb6, 0x3bb8, 0x3bba, 0x3bbb, 0x3bbd, 0x3bbe, 0x3bc0, 0x3bc1, 0x3bc3, 0x3bc4,
#             0x3bc5, 0x3bc7, 0x3bc8, 0x3bc9, 0x3bca, 0x3bcc, 0x3bcd, 0x3bce, 0x3bcf, 0x3bd0, 0x3bd1, 0x3bd2, 0x3bd3, 0x3bd4, 0x3bd5,
#             0x3bd6, 0x3bd7, 0x3bd8, 0x3bd9, 0x3bda, 0x3bdb, 0x3bdc, 0x3bdd, 0x3bdd, 0x3bde, 0x3bdf, 0x3be0, 0x3be0, 0x3be1, 0x3be2,
#             0x3be3, 0x3be3, 0x3be4, 0x3be5, 0x3be5, 0x3be6, 0x3be6, 0x3be7, 0x3be8, 0x3be8, 0x3be9, 0x3be9, 0x3bea, 0x3bea, 0x3beb,
#             0x3beb, 0x3bec, 0x3bec, 0x3bed, 0x3bed, 0x3bed, 0x3bee, 0x3bee, 0x3bef, 0x3bef, 0x3bf0, 0x3bf0, 0x3bf0, 0x3bf1, 0x3bf1,
#             0x3bf1, 0x3bf2, 0x3bf2, 0x3bf2, 0x3bf3, 0x3bf3, 0x3bf3, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf5, 0x3bf5, 0x3bf5, 0x3bf5,
#             0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf9,
#             0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfb, 0x3bfb, 0x3bfb,
#             0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc,
#             0x3bfc, 0x3bfc, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd,
#             0x3bfd, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe,
#             0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#             0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#             0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#             0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00,
#         ]
#         lut_f16 = np.array([struct.unpack('>e', struct.pack('>H', v))[0] for v in lut_hex], dtype=np.float16)
#         lut = torch.from_numpy(lut_f16)

#         # buffer NÃO persistente (não entra no state_dict)
#         self.register_buffer("lut", lut, persistent=False)

#         # ---- constantes ----
#         self.LIMIT = torch.tensor(2.0, dtype=dtype)   # limite uniforme
#         self.LUT_SIZE = lut.numel()
#         self.out_dtype = dtype

#         if device is not None:
#             self.to(device)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # garante device compatível (por segurança)
#         if self.lut.device != x.device:
#             self.lut = self.lut.to(x.device)

#         x_q = x.to(self.out_dtype)
#         abs_x = x_q.abs()

#         # mapeamento linear de |x| em [0, LIMIT] para índice [0, LUT_SIZE-1]
#         # fora do limite, clampa em LUT_SIZE-1
#         # idx = round_down( (|x|/LIMIT) * (LUT_SIZE-1) )
#         scale = (abs_x.to(torch.float32) / self.LIMIT.to(torch.float32)) * float(self.LUT_SIZE - 1)
#         idx = scale.to(torch.long).clamp(0, self.LUT_SIZE - 1)

#         # pega valor da sigmoide aproximada e aplica simetria
#         sig = self.lut[idx].to(self.out_dtype)         # \in (0.5..1] ~ metade superior
#         sig = torch.where(x_q < 0, 1.0 - sig, sig)     # simetria para x<0

#         return (x_q * sig).to(self.out_dtype)



# LUT ASYMETRIC
# class WSiLU(torch.nn.Module):
#     def __init__(self, device=None, dtype=torch.float16):
#         super().__init__()
#         # --- LUT em meia precisão (float16) a partir dos códigos hex fornecidos ---
#         lut_hex = [
#             0x3800, 0x3806, 0x380d, 0x3813, 0x381a, 0x3820, 0x3826, 0x382d, 0x3833, 0x383a, 0x3840, 0x3846,
#             0x384d, 0x3853, 0x3859, 0x3860, 0x3866, 0x386c, 0x3873, 0x3879, 0x387f, 0x3886, 0x388c, 0x3892,
#             0x3898, 0x389f, 0x38a5, 0x38ab, 0x38b1, 0x38b8, 0x38be, 0x38c4, 0x38ca, 0x38d0, 0x38d6, 0x38dc,
#             0x38e3, 0x38e9, 0x38ef, 0x38f5, 0x38fb, 0x3901, 0x3907, 0x390d, 0x3913, 0x3919, 0x391f, 0x3924,
#             0x392a, 0x3930, 0x3936, 0x393c, 0x3942, 0x3947, 0x394d, 0x3953, 0x3958, 0x395e, 0x3964, 0x3969,
#             0x396f, 0x3975, 0x397a, 0x3980, 0x3985, 0x398b, 0x3990, 0x3995, 0x399b, 0x39a0, 0x39a5, 0x39ab,
#             0x39b0, 0x39b5, 0x39ba, 0x39c0, 0x39c5, 0x39ca, 0x39cf, 0x39d4, 0x39d9, 0x39de, 0x39e3, 0x39e8,
#             0x39ed, 0x39f2, 0x39f7, 0x39fc, 0x3a01, 0x3a05, 0x3a0a, 0x3a0f, 0x3a13, 0x3a18, 0x3a1d, 0x3a21,
#             0x3a26, 0x3a2a, 0x3a2f, 0x3a33, 0x3a38, 0x3a3c, 0x3a41, 0x3a45, 0x3a49, 0x3a4e, 0x3a52, 0x3a56,
#             0x3a5a, 0x3a5f, 0x3a63, 0x3a67, 0x3a6b, 0x3a6f, 0x3a73, 0x3a77, 0x3a7b, 0x3a7f, 0x3a83, 0x3a87,
#             0x3a8a, 0x3a8e, 0x3a92, 0x3a96, 0x3a99, 0x3a9d, 0x3aa1, 0x3aa4, 0x3aa8, 0x3aac, 0x3aaf, 0x3ab3,
#             0x3ab6, 0x3ab9, 0x3abd, 0x3ac0, 0x3ac4, 0x3ac7, 0x3aca, 0x3ace, 0x3ad1, 0x3ad4, 0x3ad7, 0x3ada,
#             0x3add, 0x3ae1, 0x3ae4, 0x3ae7, 0x3aea, 0x3aed, 0x3af0, 0x3af3, 0x3af6, 0x3af8, 0x3afb, 0x3afe,
#             0x3b01, 0x3b04, 0x3b06, 0x3b09, 0x3b0c, 0x3b0f, 0x3b11, 0x3b14, 0x3b16, 0x3b19, 0x3b1c, 0x3b1e,
#             0x3b21, 0x3b23, 0x3b25, 0x3b28, 0x3b2a, 0x3b2d, 0x3b2f, 0x3b31, 0x3b34, 0x3b36, 0x3b38, 0x3b3b,
#             0x3b3d, 0x3b3f, 0x3b41, 0x3b43, 0x3b45, 0x3b47, 0x3b4a, 0x3b4c, 0x3b4e, 0x3b50, 0x3b52, 0x3b54,
#             0x3b56, 0x3b58, 0x3b5a, 0x3b5b, 0x3b5d, 0x3b5f, 0x3b61, 0x3b63, 0x3b65, 0x3b66, 0x3b68, 0x3b6a,
#             0x3b6c, 0x3b6d, 0x3b6f, 0x3b71, 0x3b72, 0x3b74, 0x3b76, 0x3b77, 0x3b79, 0x3b7a, 0x3b7c, 0x3b7e,
#             0x3b7f, 0x3b81, 0x3b82, 0x3b83, 0x3b85, 0x3b86, 0x3b88, 0x3b89, 0x3b8b, 0x3b8c, 0x3b8d, 0x3b8f,
#             0x3b90, 0x3b91, 0x3b93, 0x3b94, 0x3b95, 0x3b96, 0x3b98, 0x3b99, 0x3b9a, 0x3b9b, 0x3b9d, 0x3b9e,
#             0x3b9f, 0x3ba0, 0x3ba1, 0x3ba2, 0x3ba3, 0x3ba4, 0x3ba6, 0x3ba7, 0x3ba8, 0x3ba9, 0x3baa, 0x3bab,
#             0x3bac, 0x3bad, 0x3bae, 0x3baf, 0x3bb0, 0x3bb1, 0x3bb2, 0x3bb3, 0x3bb4, 0x3bb4, 0x3bb5, 0x3bb6,
#             0x3bb7, 0x3bb8, 0x3bb9, 0x3bba, 0x3bbb, 0x3bbb, 0x3bbc, 0x3bbd, 0x3bbe, 0x3bbf, 0x3bbf, 0x3bc0,
#             0x3bc1, 0x3bc2, 0x3bc2, 0x3bc3, 0x3bc4, 0x3bc5, 0x3bc5, 0x3bc6, 0x3bc7, 0x3bc8, 0x3bc8, 0x3bc9,
#             0x3bca, 0x3bca, 0x3bcb, 0x3bcb, 0x3bcc, 0x3bcd, 0x3bcd, 0x3bce, 0x3bcf, 0x3bcf, 0x3bd0, 0x3bd0,
#             0x3bd1, 0x3bd2, 0x3bd2, 0x3bd3, 0x3bd3, 0x3bd4, 0x3bd4, 0x3bd5, 0x3bd5, 0x3bd6, 0x3bd6, 0x3bd7,
#             0x3bd7, 0x3bd8, 0x3bd8, 0x3bd9, 0x3bd9, 0x3bda, 0x3bda, 0x3bdb, 0x3bdb, 0x3bdc, 0x3bdc, 0x3bdc,
#             0x3bdd, 0x3bdd, 0x3bde, 0x3bde, 0x3bdf, 0x3bdf, 0x3bdf, 0x3be0, 0x3be0, 0x3be1, 0x3be1, 0x3be1,
#             0x3be2, 0x3be2, 0x3be2, 0x3be3, 0x3be3, 0x3be4, 0x3be4, 0x3be4, 0x3be5, 0x3be5, 0x3be5, 0x3be6,
#             0x3be6, 0x3be6, 0x3be7, 0x3be7, 0x3be7, 0x3be7, 0x3be8, 0x3be8, 0x3be8, 0x3be9, 0x3be9, 0x3be9,
#             0x3be9, 0x3bea, 0x3bea, 0x3bea, 0x3beb, 0x3beb, 0x3beb, 0x3beb, 0x3bec, 0x3bec, 0x3bec, 0x3bec,
#             0x3bed, 0x3bed, 0x3bed, 0x3bed, 0x3bee, 0x3bee, 0x3bee, 0x3bee, 0x3bee, 0x3bef, 0x3bef, 0x3bef,
#             0x3bef, 0x3bef, 0x3bf0, 0x3bf0, 0x3bf0, 0x3bf0, 0x3bf0, 0x3bf1, 0x3bf1, 0x3bf1, 0x3bf1, 0x3bf1,
#             0x3bf2, 0x3bf2, 0x3bf2, 0x3bf2, 0x3bf2, 0x3bf2, 0x3bf3, 0x3bf3, 0x3bf3, 0x3bf3, 0x3bf3, 0x3bf3,
#             0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf5, 0x3bf5, 0x3bf5, 0x3bf5, 0x3bf5,
#             0x3bf5, 0x3bf5, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf7, 0x3bf7,
#             0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8,
#             0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9,
#             0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa,
#             0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb,
#             0x3bfb, 0x3bfb, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfe,
#             0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#             0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#         ]
#         lut_f16 = np.array([struct.unpack('>e', struct.pack('>H', v))[0] for v in lut_hex], dtype=np.float16)
#         lut = torch.from_numpy(lut_f16).to(device=device)
#         self.register_buffer("lut", lut, persistent=False)  # fica no device certo e vai junto no state_dict

#         # --- Constantes ---
#         self.DENSE_LIMIT = torch.tensor(1.5, dtype=dtype, device=device)
#         self.SPARSE_LIMIT = torch.tensor(2.0, dtype=dtype, device=device)
#         self.N_DENSE = torch.tensor(448.0, dtype=torch.float32, device=device)   # usar float p/ aritmética
#         self.N_SPARSE = torch.tensor(64.0, dtype=torch.float32, device=device)
#         self.LUT_SIZE = len(lut)
#         self.NEG_LIMIT = torch.tensor(-2.0, dtype=dtype, device=device)
#         self.out_dtype = dtype

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Tudo elemento a elemento, vetorizado
#         x_q = x.to(self.out_dtype)
#         abs_x = x_q.abs()

#         # Mapear |x| -> índice da LUT (região densa, depois esparsa; senão clampa no fim)
#         mask_dense = abs_x <= self.DENSE_LIMIT
#         mask_sparse = (abs_x > self.DENSE_LIMIT) & (abs_x <= self.SPARSE_LIMIT)

#         # Começa assumindo "fora do limite superior"
#         scaled = torch.full_like(abs_x, float(self.LUT_SIZE - 1), dtype=torch.float32)

#         # Região densa: [0, 1.5]
#         scaled = torch.where(
#             mask_dense,
#             (abs_x.to(torch.float32) / self.DENSE_LIMIT.to(torch.float32)) * self.N_DENSE,
#             scaled,
#         )

#         # Região esparsa: (1.5, 2.0]
#         scaled = torch.where(
#             mask_sparse,
#             self.N_DENSE + ((abs_x.to(torch.float32) - self.DENSE_LIMIT.to(torch.float32)) /
#                             (self.SPARSE_LIMIT.to(torch.float32) - self.DENSE_LIMIT.to(torch.float32))) * self.N_SPARSE,
#             scaled,
#         )

#         idx = scaled.to(torch.long).clamp(0, self.LUT_SIZE - 1)

#         # Sigmóide aproximada pela LUT (metade superior), com simetria para x<0
#         sig = self.lut[idx]                      # float16
#         sig = torch.where(x_q < 0, 1.0 - sig, sig)

#         # Regra especial: x < -2.0 -> 0
#         y_sigmoid = torch.where(x_q < self.NEG_LIMIT, torch.zeros_like(x_q), sig)

#         # wsilu: x * sigmoid_aproximada
#         return (x_q * y_sigmoid).to(self.out_dtype)



#DENIS COM 25 INTERVALOS E GRAU 2 (poly_25int_deg2_32)
class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        # Pontos de quebra (definem intervalos [e[i], e[i+1]))
        edges = [
            -3.0, -2.1, -1.7, -1.4, -1.1, -0.8, -0.6, -0.5, -0.4, -0.3,
            -0.2, -0.1,  0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.8,
             1.1,  1.4,  1.7,  2.1,  3.0
        ]
        # Para cada intervalo [edges[i], edges[i+1]):
        # y ≈ a0[i] + a1[i]*x + a2[i]*x^2
        a0 = [
            -0.00590918, -0.02828306, -0.06084659, -0.10340181, -0.13987062, -0.12969244,
            -0.09509441, -0.06296253, -0.03239989, -0.01105525, -0.00168324, -0.00001112,
            -0.00001112, -0.00168324, -0.01105525, -0.03239989, -0.06296253, -0.09509441,
            -0.12969244, -0.13987062, -0.10340181, -0.06084659, -0.02828306, -0.00590918
        ]
        a1 = [
            -0.00412554, -0.02532336, -0.06372518, -0.12506951, -0.19092620, -0.16258767,
            -0.04898890,  0.08031419,  0.23371924,  0.37587517,  0.46815916,  0.49880543,
             0.50119457,  0.53184084,  0.62412483,  0.76628076,  0.91968581,  1.04899094,
             1.16259233,  1.19092720,  1.12506951,  1.06372518,  1.02532336,  1.00412554
        ]
        a2 = [
            -0.00072331, -0.00575652, -0.01709311, -0.03923240, -0.06902885, -0.04948601,
             0.04392655,  0.17418195,  0.36707403,  0.60462632,  0.83360539,  0.97750736,
             0.97750736,  0.83360539,  0.60462632,  0.36707403,  0.17418195,  0.04392655,
            -0.04948601, -0.06902885, -0.03923240, -0.01709311, -0.00575652, -0.00072331
        ]

        # Registrar buffers (migram com .to(device), .half(), etc.)
        self.register_buffer("edges", torch.tensor(edges, dtype=torch.float32),persistent=False)
        self.register_buffer("a0",   torch.tensor(a0,    dtype=torch.float32),persistent=False)
        self.register_buffer("a1",   torch.tensor(a1,    dtype=torch.float32),persistent=False)
        self.register_buffer("a2",   torch.tensor(a2,    dtype=torch.float32),persistent=False)

    @torch.no_grad()  # retire se precisar de grad em x; a função é polinomial por partes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Se quiser suportar gradiente em x, remova o decorator acima;
        # o cálculo abaixo continua vetorizado do mesmo jeito.
        # Garante dtype compatível (útil em AMP/mixed precision)
        if x.dtype != self.edges.dtype:
            edges = self.edges.to(dtype=x.dtype)
            a0 = self.a0.to(dtype=x.dtype)
            a1 = self.a1.to(dtype=x.dtype)
            a2 = self.a2.to(dtype=x.dtype)
        else:
            edges, a0, a1, a2 = self.edges, self.a0, self.a1, self.a2

        # bucketize: para cada elemento de x, retorna i tal que edges[i-1] <= x < edges[i]
        idx = torch.bucketize(x, edges, right=False)

        y = torch.empty_like(x)

        # x < -3  -> 0
        left = (idx == 0)
        y[left] = 0

        # x >= 3  -> x
        right = (idx == edges.numel())
        y[right] = x[right]

        # -3 <= x < 3 -> polinômio do intervalo (idx in [1, len(edges)-1])
        mid = ~(left | right)
        if mid.any():
            ii = idx[mid] - 1                      # intervalo 0..23
            xm = x[mid]
            ym = a0[ii] + a1[ii]*xm + a2[ii]*(xm*xm)
            y[mid] = ym

        return y



#SAVE INPUTS
class WSiLU(nn.Module):
    def __init__(
        self,
        alpha: float = 4.0,
        log_path: str = "wsilu_inputs.log",
        buffer_capacity: int = 32,
        buffer_device: str | torch.device = "cuda:0",  # <<< buffer na GPU
        enable_logging: bool = True,
    ):
        """
        alpha: coeficiente do WSiLU (y = x * sigmoid(alpha * x))
        log_path: arquivo único de log (modo append binário no flush)
        buffer_capacity: número de batches acumulados antes de gravar
        buffer_device: dispositivo onde o buffer fica residente (e.g., 'cuda:0')
        enable_logging: permite desligar/ligar o log sem trocar a classe
        """
        super().__init__()
        self.alpha = alpha
        self.log_path = log_path
        self.buffer_capacity = int(buffer_capacity)
        self.buffer_device = torch.device(buffer_device)
        self.enable_logging = enable_logging

        self._buffer = []        # lista de tensores (batches) no buffer_device
        self._lock = threading.Lock()
        atexit.register(self.flush)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_logging:
            self._append_to_buffer(x)
        # WSiLU
        return x * torch.sigmoid(self.alpha * x)

    @torch.no_grad()
    def _append_to_buffer(self, x: torch.Tensor):
        # Garantir que a cópia para o buffer não cria dependência de grad
        x_buf = x.detach()
        # Mover para o dispositivo do buffer se necessário (normalmente já estará na GPU)
        if x_buf.device != self.buffer_device:
            x_buf = x_buf.to(self.buffer_device, non_blocking=True)
        with self._lock:
            self._buffer.append(x_buf)
            if len(self._buffer) >= self.buffer_capacity:
                self._flush_locked()

    @torch.no_grad()
    def flush(self):
        """Flush manual do buffer para disco."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        # Empilha no MESMO dispositivo do buffer (GPU por padrão)
        chunk_gpu = torch.stack(self._buffer, dim=0)  # shape: (N, *input_shape)
        # Opcional: você pode salvar direto o tensor CUDA. O torch.save
        # serializa corretamente o device. Isso evita transferências frequentes
        # CPU↔GPU — a transferência ocorre apenas no momento do save.
        with open(self.log_path, "ab") as f:
            torch.save(chunk_gpu, f)
        # Limpa o buffer GPU
        self._buffer.clear()

    # Utilidades
    def set_logging(self, enabled: bool):
        self.enable_logging = bool(enabled)




#WSiLU com bufferização de entrada
import atexit
import threading
import math
import json
import time
import os
from typing import Optional, Union

class WSiLU(nn.Module):
    def __init__(
        self,
        alpha: float = 4.0,
        log_path: str = "/home/ruhan/hwsigmoid_lascas2026/coding_outputs/wsilu_inputs.jsonl",
        buffer_capacity: int = 4,
        buffer_device: Union[str, torch.device] = "cuda:0",
        enable_logging: bool = True,
        sample_ratio: float = 0.01,
        min_samples: int = 1,
        random_seed: Optional[int] = None,
        log_prob: float = 0.01,  # probabilidade de adicionar um batch ao buffer
        float_precision: Optional[int] = None,  # ex.: 6 para limitar casas decimais no JSON
    ):
        """
        y = x * sigmoid(alpha * x)

        buffer_capacity: nº de batches acumulados antes de flush
        buffer_device: dispositivo do buffer (ex.: 'cuda:0')
        sample_ratio: fração do buffer gravada em cada flush (0..1)
        min_samples: mínimo de batches gravados por flush (>=0)
        random_seed: se definido, torna a amostragem reprodutível
        log_prob: probabilidade de armazenar um batch no buffer (0..1)
        float_precision: se definido, arredonda floats antes de serializar (reduz tamanho do JSON)
        """
        super().__init__()
        self.alpha = float(alpha)
        self.log_path = str(log_path)
        self.buffer_capacity = int(buffer_capacity)
        self.buffer_device = torch.device(buffer_device)
        self.enable_logging = bool(enable_logging)
        self.sample_ratio = float(sample_ratio)
        self.min_samples = int(min_samples)
        self.random_seed = random_seed
        self.log_prob = float(log_prob)
        self.float_precision = float_precision

        self._buffer = []  # lista de tensores em buffer_device
        self._lock = threading.Lock()
        atexit.register(self.flush)

        # RNG opcional para reprodutibilidade
        self._g = None
        if self.random_seed is not None:
            self._g = torch.Generator(device=self.buffer_device)
            self._g.manual_seed(self.random_seed)

        # Cria pasta do log se necessário
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_logging:
            # Decide estocasticamente se loga este batch
            if self._rand() < self.log_prob:
                self._append_to_buffer(x)
        return x * torch.sigmoid(self.alpha * x)

    # ---------- utilidades internas ----------

    def _rand(self) -> float:
        """Retorna um float em [0,1). Usa gerador próprio se existir."""
        if self._g is not None:
            return torch.rand(1, generator=self._g, device=self.buffer_device).item()
        else:
            return torch.rand(1, device=self.buffer_device).item()

    @torch.no_grad()
    def _append_to_buffer(self, x: torch.Tensor):
        x_buf = x.detach()  # remove grad
        if x_buf.device != self.buffer_device:
            x_buf = x_buf.to(self.buffer_device, non_blocking=True)
        with self._lock:
            self._buffer.append(x_buf)
            if len(self._buffer) >= self.buffer_capacity:
                self._flush_locked()

    @torch.no_grad()
    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return

        # Empilha os batches do buffer (permanece no device do buffer)
        chunk = torch.stack(self._buffer, dim=0)  # (N, *shape)
        N = chunk.shape[0]

        # Amostragem de batches: seleciona k entre N
        k = max(math.ceil(N * max(0.0, min(1.0, self.sample_ratio))), self.min_samples)
        k = min(k, N)

        if k == N:
            sampled = chunk
        else:
            if self._g is not None:
                idx = torch.randperm(N, generator=self._g, device=self.buffer_device)[:k]
            else:
                idx = torch.randperm(N, device=self.buffer_device)[:k]
            sampled = chunk.index_select(0, idx)

        # Serializa em JSONL (texto). Para isso, movemos para CPU.
        sampled_cpu = sampled.detach().to("cpu")

        # Abre em modo append texto
        with open(self.log_path, "a", encoding="utf-8") as f:
            ts = time.time()
            for i in range(sampled_cpu.shape[0]):
                entry = sampled_cpu[i]
                json_obj = self._tensor_to_json_obj(entry, ts)
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

        # Limpa buffer
        self._buffer.clear()

    def _tensor_to_json_obj(self, t: torch.Tensor, ts: float) -> dict:
        """Converte um tensor em um objeto JSON-serializável."""
        # Opcionalmente limitar precisão para reduzir tamanho
        if self.float_precision is not None and t.is_floating_point():
            t = t.round(decimals=int(self.float_precision))
        # Converte para Python list (aninhada) de forma segura
        data_list = t.tolist()

        return {
            "timestamp": ts,
            "alpha": self.alpha,
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "device_logged": str(self.buffer_device),
            "data": data_list,  # cuidado: pode ser grande!
        }

    # ---------- setters dinâmicos ----------
    def set_logging(self, enabled: bool):
        self.enable_logging = bool(enabled)

    def set_sample_ratio(self, ratio: float):
        self.sample_ratio = float(ratio)

    def set_min_samples(self, n: int):
        self.min_samples = int(n)

    def set_seed(self, seed: Optional[int]):
        self.random_seed = seed
        if seed is None:
            self._g = None
        else:
            self._g = torch.Generator(device=self.buffer_device)
            self._g.manual_seed(seed)

    def set_log_prob(self, p: float):
        self.log_prob = float(p)
