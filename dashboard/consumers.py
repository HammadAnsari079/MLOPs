import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

# Set up logging
logger = logging.getLogger(__name__)

class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        try:
            # Join room group
            await self.channel_layer.group_add(
                "dashboard_updates",
                self.channel_name
            )
            await self.accept()
            logger.info("WebSocket connection established")
        except Exception as e:
            logger.error(f"Error in WebSocket connect: {str(e)}")
            await self.close()

    async def disconnect(self, code):
        try:
            # Leave room group
            await self.channel_layer.group_discard(
                "dashboard_updates",
                self.channel_name
            )
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in WebSocket disconnect: {str(e)}")

    # Receive message from WebSocket
    async def receive(self, text_data=None, bytes_data=None):
        try:
            if text_data:
                text_data_json = json.loads(text_data)
                message = text_data_json['message']

                # Send message to room group
                await self.channel_layer.group_send(
                    "dashboard_updates",
                    {
                        'type': 'dashboard_message',
                        'message': message
                    }
                )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in WebSocket message: {str(e)}")
        except KeyError as e:
            logger.error(f"Missing key in WebSocket message: {str(e)}")
        except Exception as e:
            logger.error(f"Error in WebSocket receive: {str(e)}")

    # Receive message from room group
    async def dashboard_message(self, event):
        try:
            message = event['message']

            # Send message to WebSocket
            await self.send(text_data=json.dumps({
                'message': message
            }))
        except Exception as e:
            logger.error(f"Error in dashboard_message: {str(e)}")