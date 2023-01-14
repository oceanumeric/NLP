
## convert image to svg

```python
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
lu1 = mplt.image.imread("../images/lu-iteration.png")
ax.imshow(lu1)
ax.axis('off')
plt.savefig("../images/lu-iteration.svg", dpi=300, bbox_inches='tight')
```

## math 

\mathbb{F}

\mathbb{R}

\mathbb{C}

\mathbb{N}

```python
def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)
```