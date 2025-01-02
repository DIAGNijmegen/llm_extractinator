python -m build
python -m twine upload dist/*
rm -rf build dist *.egg-info