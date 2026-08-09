"""Модели: baseline, обучение, инференс, реестр.

Ре-экспорт здесь намеренно не делаем. `from m5.models import train` подтянул бы
LightGBM при любом обращении к пакету — включая `m5 model baseline`, которому
бустинг не нужен вовсе. Импортируем точечно: `from m5.models.baseline import ...`.
"""
