#!/usr/bin/env python3
"""Run the validation workbench on this computer only."""
import os
from pathlib import Path
import uvicorn

if __name__ == '__main__':
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.chdir(Path(__file__).resolve().parent)
    uvicorn.run('metrology.web:app', host='127.0.0.1', port=int(os.environ.get('PORT', '8765')))
