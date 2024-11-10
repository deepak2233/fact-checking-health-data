from pyngrok import ngrok

# Start a new ngrok tunnel
public_url = ngrok.connect(8000).public_url
print(f"Public URL: {public_url}")

