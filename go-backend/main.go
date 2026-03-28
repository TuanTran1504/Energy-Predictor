package main

import (
	"log"
	"os"
	"time"

	"github.com/TuanTran1504/Energy-Predictor/handlers"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	godotenv.Load("../.env")

	// Start WebSocket hub
	go handlers.GetHub().Run()

	// Start broadcast scheduler — pushes signals every 60 seconds
	go startBroadcastScheduler()

	// Start VND rate saver
	go startVNDRateSaver()

	r := gin.Default()

	r.GET("/health", handleHealth)
	r.GET("/signals", handlers.Signals)
	r.GET("/vnd/status", handlers.VNDStatus)
	r.GET("/shock/status", handlers.ShockStatus)
	r.GET("/crypto/status", handlers.CryptoStatus)
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

func startVNDRateSaver() {
	saveVNDRateOnce()
	for {
		now := time.Now().UTC()
		next := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
		time.Sleep(time.Until(next))
		saveVNDRateOnce()
	}
}

func saveVNDRateOnce() {
	result, err := handlers.FetchVNDRate()
	if err != nil {
		log.Printf("VND save failed: %v", err)
		return
	}
	log.Printf("VND rate saved: %.2f", result.USDToVND)
}
