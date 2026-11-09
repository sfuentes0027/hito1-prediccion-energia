# Certamen 2 · Hito 1 · Caso F Energía

Proyecto de demostración con datos **sintéticos** (no son consumos de una empresa real).

## Instalación
`pip install -r requirements.txt`

## Ejecutar tres pruebas
`python pipeline_energia.py --archivo datos/escenario_1_normal.csv`
`python pipeline_energia.py --archivo datos/escenario_2_alta_demanda.csv`
`python pipeline_energia.py --archivo datos/escenario_3_pico_consumo.csv`
Los JSON quedan en `resultados/`.

## Conectar n8n
1. Iniciar servidor desde esta carpeta: `uvicorn api:app --host 0.0.0.0 --port 8000`.
2. Importar `n8n/flujo_hito1_energia.json` en n8n.
3. Cambiar HOST_DEL_SCRIPT por una URL alcanzable desde n8n (localhost si está en la misma máquina y red).
4. Configurar clave del proveedor LLM de forma segura, OAuth Gmail y credenciales del bot Telegram; no subir secretos a GitHub.
5. Configurar correo destinatario y chat_id. Ejecutar tres veces, cambiando archivo del nodo HTTP y guardar capturas de ejecución, correo y mensaje.
6. Validar JSON de salida LLM y guardar registros en un repositorio/archivo persistente antes de usarlo en producción.

## Notas técnicas
El objetivo es consumo de la próxima hora. Random Forest vs baseline de persistencia (consumo de la hora actual). Split cronológico 80/20. Métricas MAE y RMSE. Los resultados son pruebas locales, **no** evidencia de envío por correo o Telegram. El flujo n8n es una plantilla que necesita configuración; no se ha ejecutado conectado.
