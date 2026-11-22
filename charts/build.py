"""Build the charts page from template.html and data.json.

Usage: python3 -I charts/build.py [artifact_out]
Writes charts/charts.html (a complete page). With an argument, also writes the bare page body used for publishing.
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, 'template.html')).read()
data = open(os.path.join(here, 'data.json')).read()
body = tpl.replace('/*DATA*/null', data)
full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
open(os.path.join(here, 'charts.html'), 'w').write(full)
if len(sys.argv) > 1:
    open(sys.argv[1], 'w').write(body)
