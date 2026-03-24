from app import app, socketio
import os

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    socketio.run(app, host='0.0.0.0', port=port)
