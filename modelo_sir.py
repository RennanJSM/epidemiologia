import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Tuple

def modelo_SIR(y: Tuple[float, float, float], t: float, N: float, beta: float, gamma: float) -> Tuple[float, float, float]:
    """  
    y: Vetor com estado atual [S, I, R]
    t: Tempo atual (não usado explicitamente nas equações, mas necessário para odeint)
    N: População total
    beta: Taxa de transmissão (contato efetivo)
    gamma: Taxa de recuperação
    
    """
    S, I, R = y
    
    # Equações diferenciais
    dSdt = -beta * S * I / N
    dIdt = (beta * S * I / N) - (gamma * I)
    dRdt = gamma * I
    
    return dSdt, dIdt, dRdt

def rodar_simulacao(N: int, I0: int, R0: int, beta: float, gamma: float, dias: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    S0 = N - I0 - R0
    y0 = (S0, I0, R0)
    
    t = np.linspace(0, dias, dias) 
    
    resultado = odeint(modelo_SIR, y0, t, args=(N, beta, gamma))
    S, I, R = resultado.T
    
    assert np.allclose(S + I + R, N, atol=1e-3), "Erro: A conservação da população foi violada!"
    
    return t, S, I, R

def plotar_resultados(t, S, I, R, nome_arquivo='grafico_epidemia.png'):
   
    plt.figure(figsize=(12, 7)) 
    
    plt.plot(t, S, label='Suscetíveis', color='blue', linewidth=2, alpha=0.8)
    plt.plot(t, I, label='Infectados', color='red', linewidth=2, alpha=0.8)
    plt.plot(t, R, label='Recuperados', color='green', linewidth=2, alpha=0.8)
    
    plt.xlabel('Tempo (dias)', fontsize=12)
    plt.ylabel('População', fontsize=12)
    plt.title(f'Simulação SIR (N={int(S[0]+I[0]+R[0])})', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    pico_dia = t[np.argmax(I)]
    pico_valor = np.max(I)
    plt.annotate(f'Pico: {int(pico_valor)} pessoas\nDia: {int(pico_dia)}', 
                 xy=(pico_dia, pico_valor), 
                 xytext=(pico_dia+20, pico_valor),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout() 
    plt.savefig(nome_arquivo, dpi=300) 
    print(f"Gráfico salvo como: {nome_arquivo}")
    plt.show()

if __name__ == "__main__":
    # 1. Definição de Parâmetros
    POPULACAO_TOTAL = 1000
    INFECTADOS_INICIAIS = 1
    RECUPERADOS_INICIAIS = 0
    TAXA_CONTAGIO_BETA = 0.3
    TAXA_RECUPERACAO_GAMMA = 0.1
    DIAS_SIMULACAO = 160

    # 2. Execução
    print("Iniciando simulação...")
    tempo, suscetiveis, infectados, recuperados = rodar_simulacao(
        N=POPULACAO_TOTAL,
        I0=INFECTADOS_INICIAIS,
        R0=RECUPERADOS_INICIAIS,
        beta=TAXA_CONTAGIO_BETA,
        gamma=TAXA_RECUPERACAO_GAMMA,
        dias=DIAS_SIMULACAO
    )

    # 3. Visualização
    plotar_resultados(tempo, suscetiveis, infectados, recuperados)
