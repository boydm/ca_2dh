# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g -O2
LDFLAGS = 
LIBS = -lraylib -lm

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Target executable
TARGET = $(BIN_DIR)/myapp

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Link the target executable
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo "Build complete: $(TARGET)"

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Clean and rebuild
rebuild: clean all

# Run the program
run: all
	./$(TARGET)

# Install (copy to /usr/local/bin)
install: all
	sudo cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	sudo rm -f /usr/local/bin/$(notdir $(TARGET))

# Debug build
debug: CFLAGS += -DDEBUG -g3
debug: clean all

# Release build
release: CFLAGS = -Wall -O3 -DNDEBUG
release: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all      - Build the project (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  rebuild  - Clean and build"
	@echo "  run      - Build and run the program"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  uninstall- Remove from /usr/local/bin"
	@echo "  debug    - Build with debug symbols"
	@echo "  release  - Build optimized release version"
	@echo "  help     - Show this help message"

# Phony targets (not actual files)
.PHONY: all clean rebuild run install uninstall debug release help directories