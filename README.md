# Карта глубины из фото (Depth Anything V2 + UI)

Локальные инструменты для генерации карт глубины из фото и совмещений, без облака.

## Компоненты

- `depth_from_photo.py` — простой конвейер: фото -> карта глубины, оверлей, DoF, сравнение "цвет + глубина". Настройки вверху файла.
- `depth_ui.py` — графический интерфейс (tkinter): выбор фото и модели, слайдеры, предпросмотр (глубина / оверлей / DoF / цветная карта / рельеф 3D / фото), крупный просмотр с зумом.
- `depth_tune.py` — интерактивный подбор настроек через вопросы.
- `depth_gen.py` — процедурный рендер сцены (сферы) с точной глубиной, без нейросети.

## Модель

Нужен ONNX Depth Anything V2 (large/small) в папке рядом:
- `depth_anything_v2_large.onnx` (~1.3 ГБ)
- `depth_anything_v2_small.onnx` (~99 МБ)

Зависимости: `onnxruntime`, `numpy`, `pillow`. venv в `depthenv/bin/python`.

## Запуск

```bash
depthenv/bin/python depth_ui.py
```

Результаты сохраняются в ту же папку (photo_depth.png, photo_depth_overlay.png, photo_dof.png, photo_color_plus_depth.png).
