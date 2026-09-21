from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AlertManager:
    cooldown_seconds: float = 5.0
    _last_alert: Dict[str, float] = field(default_factory=dict)

    def can_emit(self, class_name: str) -> bool:
        now = time.monotonic()
        last = self._last_alert.get(class_name, -1e9)
        if now - last >= self.cooldown_seconds:
            self._last_alert[class_name] = now
            return True
        return False

    @staticmethod
    def visual_message(class_name: str) -> str:
        if class_name == "Fire":
            return "🔴 ALERTA — FOGO DETECTADO"
        if class_name == "Smoke":
            return "🟠 ATENÇÃO — FUMAÇA DETECTADA"
        return "⚠️ Detecção identificada"

    def play_sound(self, class_name: str) -> None:
        if not self.can_emit(class_name):
            return
        if platform.system().lower() != "windows":
            return
        try:
            import winsound

            frequency = 1500 if class_name == "Fire" else 900
            winsound.Beep(frequency, 320)
        except Exception:
            pass
