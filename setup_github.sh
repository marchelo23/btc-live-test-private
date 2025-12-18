#!/bin/bash
# Setup script for GitHub deployment

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     SETUP GITHUB REPO - BTC Live Trading Test                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo -e "${BLUE}Inicializando repositorio git...${NC}"
    git init
    echo -e "${GREEN}✓${NC} Git inicializado"
else
    echo -e "${GREEN}✓${NC} Git ya inicializado"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Verificando archivos necesarios..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check essential files
FILES=(
    "live_predict.py"
    ".github/workflows/live_trading.yml"
    "requirements.txt"
    "README.md"
    "src/ingestion.py"
    "src/inference.py"
    "models/xgb_btc_30min_latest.json"
)

all_ok=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file - FALTA"
        all_ok=false
    fi
done

if [ "$all_ok" = false ]; then
    echo ""
    echo -e "${RED}❌ Faltan archivos necesarios${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓${NC} Todos los archivos presentes"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Preparando commit inicial..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Stage all files
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}⚠️  No hay cambios para commitear${NC}"
else
    echo -e "${BLUE}Creando commit inicial...${NC}"
    git commit -m "Initial setup: BTC live trading test system

- 5 trained models (30m, 1h, 3h, 6h, 12h)
- GitHub Actions workflow (every 30 minutes)
- CSV logging format compatible with historical system
- Average improvement: +13.63% vs historical system
"
    echo -e "${GREEN}✓${NC} Commit creado"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    PRÓXIMOS PASOS                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Crear repositorio PRIVADO en GitHub:"
echo -e "   ${BLUE}https://github.com/new${NC}"
echo "   Nombre sugerido: btc-live-test-private"
echo "   ${YELLOW}⚠️  Marcar como PRIVADO${NC}"
echo ""
echo "2. Conectar y push (reemplazar TU_USUARIO):"
echo -e "   ${BLUE}git remote add origin https://github.com/TU_USUARIO/btc-live-test-private.git${NC}"
echo -e "   ${BLUE}git branch -M main${NC}"
echo -e "   ${BLUE}git push -u origin main${NC}"
echo ""
echo "3. Configurar permisos en GitHub:"
echo "   Settings → Actions → General"
echo "   ✅ Read and write permissions"
echo ""
echo "4. Ejecutar workflow:"
echo "   Actions → Live Trading Test → Run workflow"
echo ""
echo "5. ¡Esperar y monitorear resultados!"
echo ""
echo -e "${GREEN}✅ Setup completado. Listo para deploy!${NC}"
echo ""
