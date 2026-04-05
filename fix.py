import re

with open('c:\\Users\\LENOVO\\prediction_for_restock\\main.py', 'r') as f:
    content = f.read()

# Replace the pattern
content = re.sub(r'    return predict_fn\n\n    """', '    return predict_fn\n\ndef heuristic_predict(obs):\n    """', content, 1)

with open('c:\\Users\\LENOVO\\prediction_for_restock\\main.py', 'w') as f:
    f.write(content)

print('Fixed')