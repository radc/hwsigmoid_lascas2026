import os, json, time, gzip, io
import torch
import torch.nn as nn

class TracedConv2d(nn.Conv2d):
    def __init__(self, *args,
                 track=True,
                 log_dir="logs/conv",    # pasta onde tudo será salvo
                 index_name="index.jsonl",
                 save_every=1,           # salva a cada N forwards
                 compress=True,          # salva *.pt.gz (gzip)
                 dtype_on_save=None,     # None | "float32" | "float16" | "bfloat16"
                 flush_every=20,         # flush do índice a cada N entradas
                 name=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.track = track
        self.log_dir = log_dir
        self.index_path = os.path.join(log_dir, index_name)
        self.save_every = int(save_every)
        self.compress = bool(compress)
        self.dtype_on_save = dtype_on_save
        self.flush_every = int(flush_every)
        self.name = name or "TracedConv2d"

        self._step = 0
        self._buf = []  # buffer de linhas JSONL

        os.makedirs(self.log_dir, exist_ok=True)

    # ---- utilidades de IO ----
    @staticmethod
    def _map_dtype(name, t):
        if name is None: return t
        name = str(name).lower()
        if name in ("float16","fp16"):  return t.half()
        if name in ("bfloat16","bf16"): return t.bfloat16()
        if name in ("float32","fp32"):  return t.float()
        raise ValueError(f"dtype_on_save inválido: {name}")

    def _save_tensor(self, tensor: torch.Tensor, fname: str):
        """Salva tensor (CPU, detach) em .pt ou .pt.gz, retorna caminho."""
        path = os.path.join(self.log_dir, fname + (".pt.gz" if self.compress else ".pt"))
        ten = tensor.detach().to("cpu")
        if self.dtype_on_save is not None:
            ten = self._map_dtype(self.dtype_on_save, ten)
        if self.compress:
            with gzip.open(path, "wb") as f:
                torch.save(ten, f)
        else:
            torch.save(ten, path)
        return path

    def _append_index(self, rec: dict):
        self._buf.append(json.dumps(rec))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buf: return
        with open(self.index_path, "a", encoding="utf-8") as f:
            for line in self._buf:
                f.write(line + "\n")
        self._buf.clear()

    def close(self):
        self.flush()

    # ---- forward com tracing ----
    def forward(self, x):
        y = super().forward(x)
        if not self.track:
            return y

        self._step += 1
        step = self._step

        # Sempre registramos um item no índice; salvamos arquivos só quando step % save_every == 0
        save_now = (step % self.save_every == 0)

        rec = {
            "ts": time.time(),
            "name": self.name,
            "step": step,
            "in_shape": tuple(x.shape),
            "out_shape": tuple(y.shape),
            "weight_shape": tuple(self.weight.shape),
            "bias_shape": None if self.bias is None else tuple(self.bias.shape),
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
            "saved": bool(save_now),
        }

        if save_now:
            base = f"{self.name}_step{step}"
            # salva TUDO: entrada, saída, pesos, bias
            rec["input_file"]  = self._save_tensor(x,        base + "_input")
            # rec["output_file"] = self._save_tensor(y,        base + "_output")
            # rec["weight_file"] = self._save_tensor(self.weight, base + "_weight")
            # if self.bias is not None:
            #     rec["bias_file"] = self._save_tensor(self.bias, base + "_bias")

        self._append_index(rec)
        return y
