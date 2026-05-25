CC = gcc
CFLAGS = -Wall -Wextra -D_GNU_SOURCE -I$(INCDIR) -fopenmp
SRCDIR = src
BINDIR = bin
INCDIR = include
OBJDIR = $(BINDIR)

SOURCES = $(wildcard $(SRCDIR)/*.c)
INCLUDES = $(wildcard $(INCDIR)/*.h)
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES))
TARGET = $(BINDIR)/dt_program

THREADS ?= 1
STRAETGY ?= s

all: $(BINDIR) $(TARGET)

$(BINDIR):
	mkdir -p $(BINDIR)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(INCLUDES) | $(BINDIR)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(THREADS) $(STRATEGY)

clean:
	rm -rf $(BINDIR)

.PHONY: all run clean