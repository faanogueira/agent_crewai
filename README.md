# 🤖 Agente Wikipedia com CrewAI + Ollama

<div align="center">
  <img src="agent_crewai.png" width="100%" alt="Capa do Projeto">
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/LlamaIndex-Core-7C3AED?style=flat)
![CrewAI](https://img.shields.io/badge/Google%20Gemini-2.0%20Flash-4285F4?style=flat&logo=google&logoColor=white)
![Localhost](https://img.shields.io/badge/RAG-LlamaIndex%20%2B%20Gemini-8B5CF6?style=flat)

</div>

---

---
> Agente de IA capaz de pesquisar, extrair e resumir artigos da Wikipedia de forma autônoma,  
> utilizando arquitetura multi-agente com **CrewAI** e inferência local via **Ollama**.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Execução](#execução)
- [Saída Gerada](#saída-gerada)
- [Configuração Avançada](#configuração-avançada)
- [Requisitos de Hardware](#requisitos-de-hardware)
- [Referências](#referências)

---

## Visão Geral

Este projeto implementa um sistema multi-agente com **dois agentes especializados** que colaboram para pesquisar e sintetizar conhecimento sobre **Transformers e Large Language Models (LLMs)**:

| Agente | Papel | Responsabilidade |
|---|---|---|
| 🔍 **Pesquisador** | Coleta de dados | Busca e extrai conteúdo da Wikipedia |
| ✍️ **Resumidor** | Síntese | Gera resumo estruturado em português |

A execução é totalmente **local e offline** — sem custos de API e sem envio de dados para servidores externos.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                      CrewAI                         │
│                                                     │
│   ┌─────────────────┐       ┌──────────────────┐   │
│   │  Agente 1       │  ───► │  Agente 2        │   │
│   │  Pesquisador    │       │  Resumidor       │   │
│   │                 │       │                  │   │
│   │  Tool:          │       │  Recebe contexto │   │
│   │  Wikipedia API  │       │  da Tarefa 1     │   │
│   └─────────────────┘       └──────────────────┘   │
│            │                         │              │
│            ▼                         ▼              │
│      Tarefa 1                   Tarefa 2            │
│   (Coleta de dados)         (Geração do resumo)     │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Ollama (local)  │
              │  llama3.2        │
              └──────────────────┘
```

**Fluxo de execução:** `Pesquisador busca Wikipedia` → `conteúdo passa como contexto` → `Resumidor gera síntese estruturada` → `resultado salvo em .txt`

---

## Pré-requisitos

- Python **3.10+**
- [Ollama](https://ollama.com/download) instalado
- Conexão com a internet (para baixar o modelo e acessar a Wikipedia)

---

## Instalação

### 1. Instalar o Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Baixar o modelo de linguagem

```bash
ollama pull llama3.2
```

> **Modelos alternativos disponíveis:** `mistral` · `gemma3` · `phi4`

### 3. Confirmar que o servidor Ollama está ativo

```bash
ollama serve
```

O servidor ficará disponível em `http://localhost:11434`.

> ⚠️ Se retornar `address already in use`, o Ollama já está rodando em background — pode prosseguir normalmente.

### 4. Criar e ativar o ambiente virtual Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5. Instalar as dependências

```bash
pip install crewai crewai-tools wikipedia
```

---

## Execução

```bash
python agente_wikipedia_crewai.py
```

Durante a execução, o terminal exibirá o **raciocínio de cada agente em tempo real** (`verbose=True`).  
O processo completo leva entre **3 e 10 minutos**, dependendo do hardware disponível.

---

## Saída Gerada

Ao final da execução, dois outputs são produzidos:

**1. Terminal** — exibe o resumo diretamente no console:
```
============================================================
  RESUMO FINAL GERADO PELO AGENTE
============================================================
1. O que são Transformers e LLMs
...
```

**2. Arquivo de texto** — salvo automaticamente no mesmo diretório do script:
```
resumo_transformers_llm.txt
```

Para ler o arquivo gerado:
```bash
cat resumo_transformers_llm.txt
```

---

## Configuração Avançada

### Trocar o modelo Ollama

No arquivo `agente_wikipedia_crewai.py`, localize e edite a seguinte linha:

```python
# Linha atual
llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

# Exemplos de substituição
llm = LLM(model="ollama/mistral",  base_url="http://localhost:11434")
llm = LLM(model="ollama/gemma3",   base_url="http://localhost:11434")
llm = LLM(model="ollama/phi4",     base_url="http://localhost:11434")
```

### Alterar o tema pesquisado

Na seção de tarefas do script, localize `tarefa_pesquisa` e edite a `description`:

```python
tarefa_pesquisa = Task(
    description=(
        "Use a ferramenta 'Busca na Wikipedia' para pesquisar sobre "
        "'SEU NOVO TEMA AQUI'..."  # ← edite aqui
    ),
    ...
)
```

### Alterar o idioma da Wikipedia

Na classe `WikipediaTool`, localize e edite:

```python
wikipedia.set_lang("pt")   # português (padrão)
wikipedia.set_lang("en")   # inglês
```

---

## Estrutura do Projeto

```
agent_crewai/
├── agente_wikipedia_crewai.py   # Script principal
├── resumo_transformers_llm.txt  # Gerado após a execução
├── README.md                    # Este arquivo
└── .venv/                       # Ambiente virtual Python
```

---

## Requisitos de Hardware

| Componente | Mínimo | Recomendado |
|---|---|---|
| **RAM** | 8 GB | 16 GB |
| **CPU** | Qualquer x86-64 | 8+ cores |
| **GPU** | Não obrigatória | NVIDIA com 8 GB VRAM |
| **Armazenamento** | 3 GB livres | 10 GB livres |

> 💡 Sem GPU, o modelo roda via CPU — funcional, porém mais lento.

---

## Referências

- [CrewAI — Documentação oficial](https://docs.crewai.com)
- [Ollama — Modelos disponíveis](https://ollama.com/library)
- [SentenceTransformers](https://sbert.net)
- [Wikipedia Python Library](https://pypi.org/project/wikipedia)
- [Hugging Face — paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)


---

## 👤 Autor

<!-- Início da seção "Contato" -->
<div>
  <p>Developed by <b>Fábio Nogueira</b></p>
</div>
<p>
<a href="https://www.linkedin.com/in/faanogueira/" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=13930&format=png&color=000000" target="_blank" width="80"></a>
<a href="https://github.com/faanogueira" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=AZOZNnY73haj&format=png&color=000000" target="_blank" width="80"></a>
<a href="https://api.whatsapp.com/send?phone=5571983937557" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=16713&format=png&color=000000" target="_blank" width="80"></a>
<a href="mailto:faanogueira@gmail.com"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=P7UIlhbpWzZm&format=png&color=000000" target="_blank" width="80"></a> 
</p>
<!-- Fim da seção "Contato" -->

---
