package main

import (
	"log"
	"os"
	"time"

	"github.com/TuanTran1504/Energy-Predictor/handlers"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	godotenv.Load("../.env")

	// Start WebSocket hub
	go handlers.GetHub().Run()

	// Start broadcast scheduler — pushes signals every 60 seconds
	go startBroadcastScheduler()

	r := gin.Default()
	// Add this before routes
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:5174", "http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Content-Type", "Authorization"},
		AllowWebSockets:  true,
		AllowCredentials: true,
	}))
	r.GET("/health", handleHealth)
	r.GET("/signals", handlers.Signals)
	r.GET("/shock/status", handlers.ShockStatus)
	r.GET("/crypto/status", handlers.CryptoStatus)
	r.GET("/crypto/prices", handlers.CryptoPrices)
	r.GET("/crypto/history", handlers.CryptoHistory)
	r.GET("/ml/backtest", handlers.MLBacktest)
	r.GET("/ml/market-signals", handlers.MLMarketSignals)
	r.POST("/ml/analyze", handlers.MLAnalyze)
	r.POST("/ml/trade/chat", handlers.MLTradeChat)
	r.POST("/ml/trade/execute", handlers.MLTradeExecute)
	r.GET("/macro/indicators", handlers.GetMacroIndicators)
	r.GET("/trading/live/status", handlers.LiveTradingStatus)
	r.GET("/trading/live/history", handlers.LiveTradingHistory)
	r.GET("/trading/live/fills", handlers.LiveTradeFills)
	r.GET("/ws/live", handlers.LiveFeed)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Go server starting on port %s", port)
	r.Run(":" + port)
}

func startBroadcastScheduler() {
	// Broadcast immediately on startup
	handlers.BroadcastSignals()

	// Then every 60 seconds
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		log.Println("Broadcasting signals to dashboard clients...")
		handlers.BroadcastSignals()
	}
}

func handleHealth(c *gin.Context) {
	c.JSON(200, gin.H{
		"status":  "ok",
		"service": "energy-forecaster-go",
	})
}
