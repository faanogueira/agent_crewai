# =============================================================================
# Atividade Processual 3 — Agente CrewAI com Wikipedia e Ollama
# Tema: Transformers e Large Language Models (LLMs)
#
# Arquitetura:
#   Agente 1 — Pesquisador  : busca e extrai conteúdo da Wikipedia
#   Agente 2 — Resumidor    : lê o conteúdo e gera um resumo estruturado
#
# Pré-requisitos:
#   1. Ollama instalado e rodando: https://ollama.com/download
#   2. Modelo baixado: ollama pull llama3.2
#   3. Dependências Python: pip install crewai crewai-tools wikipedia
# =============================================================================

import wikipedia
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import Field


# =============================================================================
# CONFIGURAÇÃO DO MODELO LOCAL (OLLAMA)
# =============================================================================

# O CrewAI usa o prefixo "ollama/" para identificar modelos locais via Ollama
# Certifique-se de que o Ollama está rodando: ollama serve
# Para trocar o modelo, altere "llama3.2" pelo nome do modelo instalado
# Modelos sugeridos (gratuitos): llama3.2, mistral, gemma3, phi4
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",  # Endereço padrão do servidor Ollama
)


# =============================================================================
# FERRAMENTA CUSTOMIZADA — BUSCA NA WIKIPEDIA
# =============================================================================

class WikipediaTool(BaseTool):
    """
    Ferramenta que busca e retorna o conteúdo completo de um artigo da Wikipedia.
    O agente Pesquisador usará esta ferramenta para recuperar informações sobre o tema.
    """

    name: str = "Busca na Wikipedia"
    description: str = (
        "Busca um tema na Wikipedia em português e retorna o conteúdo completo "
        "do artigo encontrado. Use esta ferramenta quando precisar pesquisar "
        "informações sobre um assunto específico. "
        "O argumento deve ser o nome do tema a pesquisar."
    )

    def _run(self, tema: str) -> str:
        """
        Executa a busca na Wikipedia e retorna o conteúdo do artigo.

        Args:
            tema: Termo a ser pesquisado na Wikipedia

        Returns:
            Conteúdo do artigo ou mensagem de erro
        """
        # Define o idioma da Wikipedia (pt = português, en = inglês)
        wikipedia.set_lang("pt")

        try:
            # Tenta buscar o artigo diretamente pelo título
            pagina = wikipedia.page(tema, auto_suggest=True)
            conteudo = pagina.content

            # Limita o conteúdo a 8000 caracteres para não sobrecarregar o LLM
            if len(conteudo) > 8000:
                conteudo = conteudo[:8000] + "\n\n[Conteúdo truncado para processamento]"

            return (
                f"TÍTULO: {pagina.title}\n"
                f"URL: {pagina.url}\n"
                f"{'=' * 60}\n"
                f"CONTEÚDO:\n{conteudo}"
            )

        except wikipedia.exceptions.DisambiguationError as e:
            # Quando há múltiplos resultados, tenta o primeiro da lista
            try:
                pagina = wikipedia.page(e.options[0])
                conteudo = pagina.content[:8000]
                return (
                    f"TÍTULO: {pagina.title}\n"
                    f"URL: {pagina.url}\n"
                    f"{'=' * 60}\n"
                    f"CONTEÚDO:\n{conteudo}"
                )
            except Exception:
                return (
                    f"Múltiplos resultados encontrados para '{tema}'. "
                    f"Opções disponíveis: {', '.join(e.options[:5])}"
                )

        except wikipedia.exceptions.PageError:
            # Tenta busca em inglês como fallback
            wikipedia.set_lang("en")
            try:
                pagina = wikipedia.page(tema, auto_suggest=True)
                conteudo = pagina.content[:8000]
                return (
                    f"[Artigo encontrado em inglês]\n"
                    f"TÍTULO: {pagina.title}\n"
                    f"URL: {pagina.url}\n"
                    f"{'=' * 60}\n"
                    f"CONTEÚDO:\n{conteudo}"
                )
            except Exception:
                return f"Artigo sobre '{tema}' não encontrado na Wikipedia."

        except Exception as e:
            return f"Erro inesperado ao buscar '{tema}': {str(e)}"


# =============================================================================
# INSTÂNCIA DA FERRAMENTA
# =============================================================================

ferramenta_wikipedia = WikipediaTool()


# =============================================================================
# AGENTE 1 — PESQUISADOR
# =============================================================================

agente_pesquisador = Agent(
    role="Pesquisador de Tecnologia de IA",
    goal=(
        "Buscar informações completas e precisas sobre Transformers e "
        "Large Language Models (LLMs) na Wikipedia, coletando dados sobre "
        "sua história, funcionamento, arquitetura e aplicações."
    ),
    backstory=(
        "Você é um pesquisador especializado em Inteligência Artificial com "
        "ampla experiência em buscar e consolidar informações técnicas de fontes "
        "confiáveis. Seu trabalho é garantir que as informações recuperadas sejam "
        "completas, precisas e relevantes para o tema solicitado."
    ),
    tools=[ferramenta_wikipedia],  # Apenas o pesquisador tem acesso à Wikipedia
    llm=llm,
    verbose=True,       # Exibe o raciocínio do agente no terminal
    memory=False,       # Desabilitado para evitar dependência de embeddings externos
    allow_delegation=False,
)


# =============================================================================
# AGENTE 2 — RESUMIDOR
# =============================================================================

agente_resumidor = Agent(
    role="Especialista em Síntese de Conhecimento",
    goal=(
        "Analisar o conteúdo pesquisado sobre Transformers e LLMs e produzir "
        "um resumo claro, estruturado e acessível, destacando os pontos mais "
        "importantes para um público técnico com interesse em ciência de dados."
    ),
    backstory=(
        "Você é um especialista em comunicação técnica com vasta experiência em "
        "transformar textos densos e complexos em resumos claros e informativos. "
        "Você possui profundo conhecimento em Inteligência Artificial e sabe "
        "identificar e destacar os conceitos mais relevantes de qualquer material."
    ),
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
)


# =============================================================================
# TAREFA 1 — PESQUISA NA WIKIPEDIA
# =============================================================================

tarefa_pesquisa = Task(
    description=(
        "Use a ferramenta 'Busca na Wikipedia' para pesquisar sobre "
        "'Transformers (aprendizado de máquina)'. "
        "Busque também por 'Large Language Model' para complementar a pesquisa. "
        "Consolide todo o conteúdo encontrado em um único documento organizado, "
        "preservando as informações técnicas sobre arquitetura, mecanismo de "
        "atenção, histórico de desenvolvimento e aplicações práticas."
    ),
    expected_output=(
        "Um documento consolidado com todo o conteúdo recuperado da Wikipedia "
        "sobre Transformers e LLMs, incluindo: definição, histórico, "
        "arquitetura técnica, mecanismo de atenção (attention mechanism), "
        "modelos conhecidos (BERT, GPT, etc.) e aplicações."
    ),
    agent=agente_pesquisador,
)


# =============================================================================
# TAREFA 2 — GERAÇÃO DO RESUMO
# =============================================================================

tarefa_resumo = Task(
    description=(
        "Com base no conteúdo pesquisado pelo Pesquisador, produza um resumo "
        "estruturado e completo sobre Transformers e Large Language Models. "
        "O resumo deve ser escrito em português e conter as seguintes seções:\n\n"
        "1. O que são Transformers e LLMs\n"
        "2. Breve histórico e surgimento\n"
        "3. Como funcionam (arquitetura e mecanismo de atenção)\n"
        "4. Principais modelos conhecidos\n"
        "5. Aplicações práticas no mundo real\n"
        "6. Impacto e perspectivas futuras\n\n"
        "Use linguagem técnica mas acessível, adequada para profissionais "
        "em transição para a área de ciência de dados."
    ),
    expected_output=(
        "Um resumo estruturado em português, com as 6 seções solicitadas, "
        "cada uma com pelo menos 2 parágrafos explicativos. "
        "O texto deve ser claro, técnico e informativo, com aproximadamente "
        "600 a 900 palavras no total."
    ),
    agent=agente_resumidor,
    context=[tarefa_pesquisa],  # Esta tarefa depende do resultado da tarefa 1
)


# =============================================================================
# CREW — ORQUESTRA OS AGENTES E TAREFAS
# =============================================================================

crew = Crew(
    agents=[agente_pesquisador, agente_resumidor],
    tasks=[tarefa_pesquisa, tarefa_resumo],
    process=Process.sequential,  # Executa tarefa 1 antes da tarefa 2
    verbose=True,
)


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AGENTE WIKIPEDIA — CrewAI + Ollama")
    print("  Tema: Transformers e Large Language Models")
    print("=" * 60 + "\n")

    # Inicia a execução da crew
    resultado = crew.kickoff()

    # Exibe o resultado final
    print("\n" + "=" * 60)
    print("  RESUMO FINAL GERADO PELO AGENTE")
    print("=" * 60)
    print(resultado)

    # Salva o resultado em um arquivo de texto
    with open("resumo_transformers_llm.txt", "w", encoding="utf-8") as f:
        f.write("RESUMO: TRANSFORMERS E LARGE LANGUAGE MODELS\n")
        f.write("Gerado por: CrewAI + Ollama (llama3.2)\n")
        f.write("Fonte: Wikipedia\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(resultado))

    print("\n✅ Resumo salvo em: resumo_transformers_llm.txt")
