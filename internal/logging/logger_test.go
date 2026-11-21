package logging

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDualLogger(t *testing.T) {
	dir := t.TempDir()
	lg, err := NewDualLogger(dir, "status.log", "processing.log", "INFO", false)
	if err != nil {
		t.Fatalf("NewDualLogger failed: %v", err)
	}
	lg.Debug("test", "ignored debug")
	lg.Status("DOWNLOAD", "START", "starting download %d", 1)
	lg.Info("loader", "loaded %d rows", 100)
	lg.Error("trainer", "sample error %s", "boom")
	if err := lg.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	statusBytes, err := os.ReadFile(filepath.Join(dir, "status.log"))
	if err != nil {
		t.Fatalf("read status.log: %v", err)
	}
	procBytes, err := os.ReadFile(filepath.Join(dir, "processing.log"))
	if err != nil {
		t.Fatalf("read processing.log: %v", err)
	}

	statusStr := string(statusBytes)
	procStr := string(procBytes)
	if !strings.Contains(statusStr, "[DOWNLOAD] [START] starting download 1") {
		t.Fatalf("unexpected status.log content: %s", statusStr)
	}
	if strings.Contains(procStr, "ignored debug") {
		t.Fatalf("DEBUG message should have been filtered at INFO level")
	}
	if !strings.Contains(procStr, "[INFO] [loader] loaded 100 rows") ||
		!strings.Contains(procStr, "[ERROR] [trainer] sample error boom") {
		t.Fatalf("unexpected processing.log content: %s", procStr)
	}
}
