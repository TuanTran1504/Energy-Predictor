package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// upgrader converts an HTTP connection to a WebSocket connection
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

type Client struct {
	conn *websocket.Conn
	send chan []byte
}
type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.RWMutex
}

var hub = &Hub{
	clients:    make(map[*Client]bool),
	broadcast:  make(chan []byte, 256),
	register:   make(chan *Client),
	unregister: make(chan *Client),
}

func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
			fmt.Printf("Client connected. Total: %d\n", len(h.clients))

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				func() {
					defer func() {
						if r := recover(); r != nil {
							fmt.Printf("Recovered from close on closed channel: %v\n", r)
						}
					}()
					close(client.send)
				}()
			}
			h.mu.Unlock()
			fmt.Printf("Client disconnected. Total: %d\n", len(h.clients))

		case message := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				select {
				case client.send <- message:

				default:

					h.mu.RUnlock()
					h.unregister <- client
					h.mu.RLock()
				}
			}
			h.mu.RUnlock()
		}
	}
}
func GetHub() *Hub {
	return hub
}

// writePump pumps messages from the hub to the WebSocket connection
func (c *Client) writePump() {
	ticker := time.NewTicker(30 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				// Hub closed the channel
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}

		case <-ticker.C:
			// Ping every 30 seconds to keep connection alive
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (c *Client) readPump() {
	defer func() {
		hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(512)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, _, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err,
				websocket.CloseGoingAway,
				websocket.CloseAbnormalClosure) {
				fmt.Printf("WebSocket error: %v\n", err)
			}
			break
		}
	}
}

// LiveFeed handles GET /ws/live — dashboard connects here
func LiveFeed(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		fmt.Printf("WebSocket upgrade failed: %v\n", err)
		return
	}

	client := &Client{
		conn: conn,
		send: make(chan []byte, 256),
	}

	hub.register <- client

	// Send current signals immediately on connect
	// so the dashboard has data before the next broadcast
	go func() {
		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("Recovered from panic in initial send: %v\n", r)
			}
		}()

		payload := buildSignalPayload()
		if data, err := json.Marshal(payload); err == nil {
			select {
			case client.send <- data:
				// Successfully sent
			default:
				// Channel buffer full or closed — skip
			}
		}
	}()

	// Start read and write pumps in separate goroutines
	go client.writePump()
	client.readPump() // blocks until client disconnects
}

// BroadcastSignals pushes latest signals to all connected clients
// Called by the scheduler every 60 seconds
func BroadcastSignals() {
	payload := buildSignalPayload()
	data, err := json.Marshal(payload)
	if err != nil {
		fmt.Printf("Error marshaling broadcast: %v\n", err)
		return
	}
	hub.broadcast <- data
}

// WebSocketMessage is the shape of every message sent to the dashboard
type WebSocketMessage struct {
	Type          string                   `json:"type"`
	Timestamp     string                   `json:"timestamp"`
	Signals       SignalsResponse          `json:"signals"`
	ClientCount   int                      `json:"client_count"`
	MLPredictions map[string]*MLPrediction `json:"ml_predictions,omitempty"`
}

func buildSignalPayload() WebSocketMessage {
	brentCh := make(chan *BrentResult, 1)
	cryptoCh := make(chan *CryptoSignals, 1)
	mlCh := make(chan map[string]*MLPrediction, 1)

	go func() {
		delta, err := fetchOilDeltaPct()
		if err != nil {
			brentCh <- &BrentResult{PriceUSD: 104.20, DeltaDayPct: 0}
			return
		}
		brentCh <- &BrentResult{PriceUSD: 104.20, DeltaDayPct: delta}
	}()

	go func() {
		signals, err := FetchCryptoSignals()
		if err != nil {
			cryptoCh <- &CryptoSignals{}
			return
		}
		cryptoCh <- signals
	}()

	go func() {
		mlCh <- FetchMLPredictions()
	}()

	brent := <-brentCh
	crypto := <-cryptoCh
	mlPreds := <-mlCh

	hub.mu.RLock()
	clientCount := len(hub.clients)
	hub.mu.RUnlock()

	return WebSocketMessage{
		Type:      "signals_update",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Signals: SignalsResponse{
			BrentCrude: *brent,
			Crypto:     *crypto,
			FetchedAt:  time.Now().UTC().Format(time.RFC3339),
		},
		ClientCount:   clientCount,
		MLPredictions: mlPreds,
	}
}
