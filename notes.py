import numpy as np
import matplotlib.pyplot as plt

# Definizione funzione
def g(x, y):
    return 0.5 * (np.sqrt((1+x)*(1+y)) + np.sqrt((1-x)*(1-y)))

def h(x,y):
    return np.sqrt((1-x)*(1-y))-np.sqrt(x*y)

# Dominio (la funzione è reale per x,y in [-1,1])
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
E1, E2 = np.meshgrid(x, y)
Z1 = g(E1, E2)
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
H1, H2 = np.meshgrid(x, y)
Z2 = h(H1,H2)

# Figura con due subplot 3D
fig = plt.figure(figsize=(12,5))

# --- Primo grafico ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(E1, E2, Z1, cmap='viridis')
ax1.set_title('g surface over Ex domain')
ax1.set_xlabel('E1')
ax1.set_ylabel('E2')
ax1.set_zlabel('g(E1,E2)')
cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1)
# --- Secondo grafico ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(H1, H2, Z2, cmap='plasma')
ax2.set_title('h surface over Hx domain')
ax2.set_xlabel('H1')
ax2.set_ylabel('H2')
ax2.set_zlabel('h(H1,H2)')
cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1)
plt.tight_layout()
plt.show()
plt.show()