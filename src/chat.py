from search import search_prompt


def main():
    try:
        print("Inicializando chat...")
        chain = search_prompt()
        if not chain:
            print("❌ Não foi possível inicializar o chat. Verifique os erros de inicialização.")
            return
    except Exception as e:
        print(f"❌ Erro ao inicializar o chat: {e}")
        print("Certifique-se de que:")
        print("  1. Docker está rodando (docker compose up -d)")
        print("  2. O arquivo .env está configurado com API keys válidas")
        print("  3. Você executou python src/ingest.py primeiro")
        return

    print("✅ Chat inicializado com sucesso!")
    print("Digite 'sair' ou use Ctrl+C para encerrar.\n")

    while True:
        try:
            print("Faça sua pergunta:")
            question = input("> ").strip()

            if not question:
                continue

            if question.lower() in ["sair", "quit", "exit"]:
                print("Até logo!")
                break

            print("\nProcessando...")
            response = search_prompt(question)
            print(f"\nRESPOSTA: {response}\n")
            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\nEncerrando chat...")
            break
        except Exception as e:
            print(f"❌ Erro ao processar pergunta: {e}")
            print("Tente novamente.\n")


if __name__ == "__main__":
    main()