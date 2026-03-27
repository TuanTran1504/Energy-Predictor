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

	// Start background VND rate saver
	go startVNDRateSaver()

	r := gin.Default()

	r.GET("/health", handleHealth)
	r.GET("/shock/status", handlers.ShockStatus)
	r.POST("/forecast", handlers.Forecast)
	r.GET("/vnd/status", handlers.VNDStatus)
	r.GET("/signals", handlers.Signals)
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Go server starting on port %s", port)
	r.Run(":" + port)
}

func handleHealth(c *gin.Context) {
	c.JSON(200, gin.H{
		"status":  "ok",
		"service": "energy-forecaster-go",
	})
}

// startVNDRateSaver runs in background — saves one row per day at midnight UTC
// Now that we have 2 years of history backfilled, this just adds one row per day
func startVNDRateSaver() {
	// Save once immediately on startup
	saveVNDRateOnce()

	// Then save once per day at midnight UTC
	for {
		now := time.Now().UTC()
		next := time.Date(
			now.Year(), now.Month(), now.Day()+1,
			0, 0, 0, 0, time.UTC,
		)
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
	log.Printf("VND rate saved automatically: %.2f", result.USDToVND)
}
