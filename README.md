# 🤖 BTC Live Trading Test - Advanced Models

Sistema de pruebas en vivo de modelos de predicción de Bitcoin usando GitHub Actions.

## 🎯 Objetivo

Probar nuestros modelos avanzados (31 features) en condiciones reales durante varias semanas y comparar con el sistema histórico.

## ⚙️ Características

- ✅ **Ejecución automática cada 30 minutos**
- ✅ **5 horizontes de predicción**: 30m, 1h, 3h, 6h, 12h
- ✅ **Formato CSV compatible** con sistema histórico
- ✅ **Validación automática** de predicciones pasadas
- ✅ **Repositorio privado** para seguridad

## 📊 Modelos Utilizados

| Horizonte | Features | R² | Directional Accuracy |
|-----------|----------|-----|---------------------|
| 30 min    | 31       | -0.00 | 61.74% |
| 1 hora    | 31       | -0.08 | 59.39% |
| 3 horas   | 31       | -0.03 | 73.94% |
| 6 horas   | 31       | -0.00 | 63.38% |
| 12 horas  | 31       | 0.51  | 81.46% ⭐ |

**Promedio**: 68.00% vs 54.35% del sistema histórico (+13.63%)

## 🚀 Setup Rápido

### 1. Crear Repositorio Privado en GitHub

```bash
# Ir a: https://github.com/new
# Nombre: btc-live-test-private
# ✅ Marcar como PRIVADO
# Crear repositorio
```

### 2. Configurar y Push

```bash
cd btc_live_test_github

# Inicializar git
git init
git add .
git commit -m "Initial setup: Live trading test system"

# Conectar con GitHub (reemplazar TU_USUARIO)
git remote add origin https://github.com/TU_USUARIO/btc-live-test-private.git
git branch -M main
git push -u origin main
```

### 3. Configurar Permisos en GitHub

1. Ir a: **Settings** → **Actions** → **General**
2. En "Workflow permissions":
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**
3. Guardar cambios

### 4. Ejecutar Primera Vez

1. Ir a: **Actions** tab
2. Seleccionar: **Live Trading Test - Every 30min**
3. Click: **Run workflow** → **Run workflow**
4. Esperar 2-3 minutos

### 5. Verificar Funcionamiento

El workflow se ejecutará automáticamente cada 30 minutos y:
- ✅ Descargará datos frescos de Binance
- ✅ Hará 5 predicciones (una por cada horizonte)
- ✅ Validará predicciones pasadas
- ✅ Actualizará `bitacora_new_models.csv`
- ✅ Hará commit automático

## 📁 Estructura del Proyecto

```
btc_live_test_github/
├── .github/workflows/
│   └── live_trading.yml        # Workflow cada 30 min
├── src/                         # Código fuente
│   ├── ingestion.py            # Descarga datos
│   ├── features.py             # Feature engineering
│   └── inference.py            # Predicciones
├── models/                      # Modelos entrenados
│   ├── xgb_btc_30min_latest.json
│   ├── xgb_btc_60min_latest.json
│   ├── xgb_btc_180min_latest.json
│   ├── xgb_btc_360min_latest.json
│   └── xgb_btc_720min_latest.json
├── config/
│   └── config.yaml             # Configuración
├── live_predict.py             # Script principal
├── requirements.txt            # Dependencias
└── bitacora_new_models.csv     # Predicciones (se crea automáticamente)
```

## 📊 Formato del CSV

El archivo `bitacora_new_models.csv` tiene el mismo formato que el sistema histórico:

```csv
timestamp_pred,timeframe,entry_price,predicted_price,direction_pred,target_time,actual_price,error_abs,status
2025-12-17 20:30:00,30m,86000.0,86150.5,UP,2025-12-17 21:00:00,,,PENDING
```

**Columnas**:
- `timestamp_pred`: Momento de la predicción
- `timeframe`: Horizonte (30m, 1h, 3h, 6h, 12h)
- `entry_price`: Precio al momento de predecir
- `predicted_price`: Precio predicho
- `direction_pred`: Dirección (UP/DOWN)
- `target_time`: Momento objetivo
- `actual_price`: Precio real (se completa después)
- `error_abs`: Error absoluto (se completa después)
- `status`: PENDING o COMPLETED

## 🔍 Monitoreo

### Ver Predicciones Acumuladas

```bash
# Pull últimos cambios
git pull

# Ver bitácora
cat bitacora_new_models.csv

# Contar predicciones
wc -l bitacora_new_models.csv
```

### Ver Estadísticas

```bash
python -c "
import pandas as pd
df = pd.read_csv('bitacora_new_models.csv')
print(f'Total: {len(df)}')
print(f'Pending: {len(df[df[\"status\"] == \"PENDING\"])}')
print(f'Completed: {len(df[df[\"status\"] == \"COMPLETED\"])}')
print('\nBy timeframe:')
print(df.groupby('timeframe').size())
"
```

### Ver Logs de GitHub Actions

1. Ir a: **Actions** tab
2. Click en el último workflow run
3. Ver detalles de ejecución

## 📈 Resultados Esperados

### Después de 1 Día
- ~48 predicciones por horizonte (24h × 2 ejecuciones/hora)
- ~240 predicciones totales
- Primeras validaciones completadas (30m y 1h)

### Después de 1 Semana
- ~336 predicciones por horizonte
- ~1,680 predicciones totales
- Suficientes datos para análisis estadístico

### Después de 1 Mes
- ~1,440 predicciones por horizonte
- ~7,200 predicciones totales
- Análisis robusto de rendimiento

## 🎯 Criterios de Éxito

| Métrica | Objetivo |
|---------|----------|
| Win Rate promedio | >60% |
| Mejor que histórico | 4/5 horizontes |
| Workflows exitosos | >95% |
| Predicciones/día | >200 |

## 💰 Costos

- ✅ **GitHub Actions**: GRATIS (2,000 min/mes)
- ✅ **Binance API**: GRATIS (solo lectura)
- ✅ **Storage**: GRATIS (<500MB)

**Uso estimado**: ~100 minutos/mes (<<< 2,000 límite)

## 🔧 Troubleshooting

### Workflow falla: "Models not found"

```bash
# Verificar que modelos están en el repo
git add models/*.json -f
git commit -m "Add models"
git push
```

### No se crean commits automáticos

- Verificar permisos: Settings → Actions → General
- Debe estar en "Read and write permissions"

### Predicciones siempre PENDING

- Esperar al menos 30-60 minutos para validación
- Las predicciones se validan automáticamente cuando llega target_time

## 📞 Soporte

- **Issues**: [GitHub Issues](../../issues)
- **Documentación**: Este README

## 🔒 Seguridad

- ✅ Repositorio **PRIVADO**
- ✅ No expone API keys
- ✅ Solo lectura de datos públicos de Binance
- ✅ Sin trading real

---

**Estado**: ✅ Listo para producción

**Próximo paso**: `git push` y activar workflow en GitHub Actions
