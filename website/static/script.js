// --- START OF FULL UPDATED script.js ---
document.addEventListener("DOMContentLoaded", function () {
    const gameBoard = document.getElementById('gameBoard');
    const rows = 13;
    const cols = 13;
    const hexWidth = 40;
    const marginIncrement = hexWidth * 0.5 + 2;

    gameBoard.style.setProperty('--hex-width', `${hexWidth}px`);
    gameBoard.style.setProperty('margin-top', `100px`);

    let letters = 'ABCDEFGHIJK'; // For 11x11 board
    let moves = []; // Stores the full history of the current game path
    let currentMoveIndex = 0; // Index of the *next* move to be made (or current number of moves displayed)
    let isGameOver = false; // Flag for game completion
    const _allow_swap_rule = true; // <<< Enable/Disable Swap Rule

    // Internal board representation 11x11
    const boardSize = 11;
    const board = Array.from(Array(boardSize), () => Array(boardSize).fill(0)); // 0: empty, 1: red, 2: blue

    // Generate visual game board
    for (let i = 0; i < rows; i++) {
        const row = document.createElement('div');
        row.classList.add('hex-row');
        row.style.marginLeft = i * marginIncrement + 'px';
        for (let j = 0; j < cols; j++) {
            let hex;
            if ((i === 0 && j === 12)) {
                hex = document.createElement('div'); hex.classList.add('hex_special1');
            } else if (i === 12 && j === 0) {
                hex = document.createElement('div'); hex.classList.add('hex_special2');
            } else {
                hex = document.createElement('div'); hex.classList.add('hexagon');
                if ((i === 0 && j === 0) || (i === 0 && j === cols - 1) || (i === rows - 1 && j === 0) || (i === rows - 1 && j === cols - 1)) {
                    hex.classList.add('hex_corner');
                } else if (i === 0 || i === rows - 1) {
                    hex.classList.add('hex_border', 'horizontal'); // Red borders
                } else if (j === 0 || j === cols - 1) {
                    hex.classList.add('hex_border', 'vertical'); // Blue borders
                } else {
                    hex.classList.add('hex_playable');
                    const letter = letters[j - 1];
                    const number = i;
                    hex.id = `${letter}${number}`; // Assign ID like A1, K11
                }
            }
            row.appendChild(hex);
        }
        gameBoard.appendChild(row);
    }

    // --- Get DOM Elements ---
    const textInput = document.getElementById('textInput');
    const hoverInfo = document.getElementById('hoverInfo');
    const gameLog = document.getElementById('gameLog');
    const currentMoveDisplay = document.getElementById('currentMove');
    const postButton = document.getElementById('postButton');
    const getButton = document.getElementById('getButton');
    const prevMoveButton = document.getElementById('prevMove');
    const nextMoveButton = document.getElementById('nextMove');
    const newGameButton = document.getElementById('newGameButton');
    const aiBluePlaysToggle = document.getElementById('aiBluePlaysToggle');
    const aiRedPlaysToggle = document.getElementById('aiRedPlaysToggle');

    // --- Utility Functions ---

    function isValidMove(move, size = boardSize) {
        if (!move || move.length < 2 || move.length > 3) return false;
        const letter = move[0].toUpperCase();
        if (letter < 'A' || letter > String.fromCharCode('A'.charCodeAt(0) + size - 1)) return false;
        const numberStr = move.slice(1);
        const number = parseInt(numberStr);
        return !isNaN(number) && number >= 1 && number <= size && number.toString() === numberStr;
    }

    function convertMoveToCoords(move) {
        const letter = move[0].toUpperCase();
        const number = parseInt(move.slice(1));
        const x = letter.charCodeAt(0) - 'A'.charCodeAt(0); // Column index (0-10)
        const y = number - 1; // Row index (0-10)
        return [x, y];
    }

    function coordsToMove(x, y, size = boardSize) {
        if (x < 0 || x >= size || y < 0 || y >= size) {
            console.error(`Invalid coordinates for coordsToMove: [${x}, ${y}]`);
            return null;
        }
        const letter = String.fromCharCode('A'.charCodeAt(0) + x);
        const number = y + 1;
        return `${letter}${number}`;
    }

    function isCellOccupied(move) {
        // Checks the INTERNAL board state, used mainly by AI before making a move
        if (!isValidMove(move)) return true;
        const [x, y] = convertMoveToCoords(move);
        if (y < 0 || y >= boardSize || x < 0 || x >= boardSize) return true;
        return board[y][x] !== 0;
    }

    // --- Event Listeners ---

    textInput.addEventListener('input', function (e) {
        textInput.value = textInput.value.replace(/[^A-Ka-k1-9 ]|(1[01])?/g, '').replace(/\s{2,}/g, ' ').toUpperCase();
    });

    postButton.addEventListener('click', function () {
        const inputMovesRaw = textInput.value.trim().split(' ');
        const validMoves = [];
        // Need to properly handle swap if present in text input
        // Simple validation for now, does NOT interpret swap from text
        const tempUniqueCheck = new Set();
        for (let i = 0; i < inputMovesRaw.length; i++) {
             const move = inputMovesRaw[i];
             if (isValidMove(move)) {
                 // Basic duplicate check within the input sequence itself
                 if (!tempUniqueCheck.has(move)) {
                    validMoves.push(move);
                    tempUniqueCheck.add(move);
                 } else {
                     // Allow duplicate *only* if it's the second move and matches the first (potential swap)
                     // More robust text parsing would be needed to fully handle swap input via text.
                    if (!(i === 1 && _allow_swap_rule && move === inputMovesRaw[0])) {
                         console.warn(`Duplicate move ignored in text input: ${move}`);
                     } else {
                         // If it *looks* like a swap input, keep it for applyMovesToBoard to decide
                         validMoves.push(move);
                     }
                 }
             } else if (move) {
                 console.warn(`Invalid move format ignored in text input: ${move}`);
             }
        }

        moves = validMoves;
        currentMoveIndex = moves.length;
        isGameOver = false;
        displayBoardState(currentMoveIndex); // applyMovesToBoard will handle swap logic based on moves array
        checkAITurn();
    });


    getButton.addEventListener('click', function () {
        textInput.value = moves.slice(0, currentMoveIndex).join(' ');
    });

    // --- Hex Click Listener (Handles Player Moves and SWAP) ---
    document.querySelectorAll('.hex_playable').forEach(hex => {
        let timeout;
        hex.addEventListener('mouseenter', function () {
            timeout = setTimeout(() => { hoverInfo.innerText = hex.id; hoverInfo.classList.add('show'); }, 10);
        });
        hex.addEventListener('mouseleave', function () {
            clearTimeout(timeout); hoverInfo.classList.remove('show');
        });

        hex.addEventListener('click', function () {
            if (isGameOver) return;

            const clickedMoveId = hex.id;
            let moveToAdd = clickedMoveId;
            let isSwapAttempt = false;

            // --- SWAP RULE TRIGGER LOGIC ---
            if (_allow_swap_rule && currentMoveIndex === 1 && moves.length === 1 && clickedMoveId === moves[0]) {
                isSwapAttempt = true;
                const [x0, y0] = convertMoveToCoords(moves[0]);
                const swappedX = y0;
                const swappedY = x0;
                const swappedMoveId = coordsToMove(swappedX, swappedY);

                if (!swappedMoveId) {
                    console.error("Error calculating swapped move ID. Aborting move.");
                    return;
                }

                // !!! REMOVED OCCUPANCY CHECK HERE !!!
                // We *expect* the clicked cell (moves[0]) to be occupied by Red.
                // We *trust* applyMovesToBoard to handle clearing Red's stone
                // and placing Blue's stone correctly on the swapped target.

                console.log(`SWAP RULE TRIGGERED! Red played ${moves[0]}. Blue clicked ${clickedMoveId}. Swapping to ${swappedMoveId}.`);
                moveToAdd = swappedMoveId; // Use the swapped move ID for history
            }
            // --- END SWAP RULE TRIGGER LOGIC ---
            else {
                 // Normal Move Attempt: Check if the clicked cell is visually occupied
                if (hex.classList.contains('red') || hex.classList.contains('blue')) {
                     console.log("Cell already occupied (visually).");
                     return; // Don't add move if cell visually occupied
                }
            }

            // Add the determined move (original or swapped) to history and update state
            addMove(moveToAdd);
            checkAITurn(); // Check if AI needs to move next
        });
    });


    prevMoveButton.addEventListener('click', function () {
        if (currentMoveIndex > 0) {
            isGameOver = false;
            currentMoveIndex--;
            displayBoardState(currentMoveIndex);
        }
    });

    nextMoveButton.addEventListener('click', function () {
        if (currentMoveIndex < moves.length) {
            currentMoveIndex++;
            displayBoardState(currentMoveIndex);
        }
    });

    newGameButton.addEventListener('click', startNewGame);
    aiBluePlaysToggle.addEventListener('change', checkAITurn);
    aiRedPlaysToggle.addEventListener('change', checkAITurn);

    // --- Log Click Listener ---
    gameLog.addEventListener('click', function(event) {
        const targetSpan = event.target.closest('.log-move');
        if (targetSpan && targetSpan.dataset.moveIndex !== undefined) {
            const index = parseInt(targetSpan.dataset.moveIndex, 10);
            if (!isNaN(index) && index >= 0 && index < moves.length) {
                currentMoveIndex = index + 1;
                isGameOver = false;
                displayBoardState(currentMoveIndex); // applyMoves + visual update
                console.log(`Navigated via log to move index ${index} (State after move ${currentMoveIndex})`);
            }
        }
    });

    // --- Core Game Logic Functions ---

    function addMove(move) {
        if (isGameOver) return;
        if (!isValidMove(move)) {
             console.error(`Attempted to add invalid move format: ${move}`);
             return;
        }

        if (currentMoveIndex < moves.length) {
            moves = moves.slice(0, currentMoveIndex);
        }

        moves.push(move);
        currentMoveIndex++;
        displayBoardState(currentMoveIndex); // Central update function
    }

    // Updates visuals, internal state, counter, log, checks victory
    function displayBoardState(moveIndexToShow) {
        // 1. Update internal board state based on history up to this point
        applyMovesToBoard(moves.slice(0, moveIndexToShow));

        // 2. Clear visual board colors
        document.querySelectorAll('.hex_playable').forEach(hex => {
            hex.classList.remove('red', 'blue');
        });

        // 3. Apply colors visually based *only* on the final internal board state
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                const moveId = coordsToMove(c, r);
                if (moveId) {
                    const hex = document.getElementById(moveId);
                    if (hex) {
                        if (board[r][c] === 1) {
                            hex.classList.add('red');
                        } else if (board[r][c] === 2) {
                            hex.classList.add('blue');
                        }
                    }
                }
            }
        }

        // 4. Update current move counter
        currentMoveDisplay.textContent = moveIndexToShow;
        // currentMoveIndex is already set by caller

        // 5. Update log and highlighting
        displayLog(); // Show moves up to currentMoveIndex

        // 6. Check for victory based on the *current* internal board state
        if (!isGameOver) { // Avoid re-checking if already over
            checkVictory();
        }

        // 7. Disable/Enable history buttons
        prevMoveButton.disabled = currentMoveIndex === 0;
        // Disable next if at the end OR if the game is over
        nextMoveButton.disabled = currentMoveIndex === moves.length || isGameOver;
    }


    // --- APPLY MOVES TO BOARD (Handles Swap Rule Internally) ---
    function applyMovesToBoard(movesToApply) {
        // 1. Reset internal board
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                board[r][c] = 0;
            }
        }

        // 2. Apply moves sequentially
        for (let i = 0; i < movesToApply.length; i++) {
            const move = movesToApply[i]; // The move string (e.g., "A1", "C3") from history

             if (!isValidMove(move)) {
                 console.error(`Invalid move encountered during applyMovesToBoard: ${move} at index ${i}`);
                 continue;
            }

            const [x, y] = convertMoveToCoords(move); // Coords for the move in history

            if (y < 0 || y >= boardSize || x < 0 || x >= boardSize) {
                 console.error(`Coords out of bounds for move: ${move} ([${x}, ${y}]) at index ${i}`);
                 continue;
             }

            // --- SWAP RULE LOGIC within board application ---
            // Check if it's the second move (i=1) and swap is allowed
            if (i === 1 && _allow_swap_rule && movesToApply.length > 0) {
                const firstMoveActual = movesToApply[0]; // Red's actual first move string
                const [x0, y0] = convertMoveToCoords(firstMoveActual); // Red's actual first coords

                // Determine if the *second move recorded in history* (move = movesToApply[1])
                // corresponds to a swap action. This is true IF its coordinates [x, y]
                // are the coordinate-swap of Red's first move's coordinates [x0, y0].
                if (x === y0 && y === x0) {
                    // SWAP OCCURRED based on history analysis!
                    // a) Ensure Red's original stone position [x0, y0] is empty on the board.
                    //    (It should be 0 because we reset the board, but good practice)
                    if (board[y0][x0] !== 0) {
                         console.warn(`Swap Rule Anomaly: Red's original cell ${firstMoveActual} wasn't 0 before clearing.`);
                         board[y0][x0] = 0; // Force clear just in case
                    }
                    // b) Place Blue's stone (player 2) at the swapped location [x, y] = [y0, x0]
                    if (board[y][x] === 0) { // Ensure target is clear before placing
                        board[y][x] = 2; // Place Blue
                        console.log(`ApplyMoves: SWAP applied. Cleared ${firstMoveActual} ([${x0},${y0}]). Blue placed at ${move} ([${x},${y}])`);
                    } else {
                        // This should not happen if move validation/logic is correct
                        console.error(`ApplyMoves SWAP Error: Target cell ${move} ([${x},${y}]) was unexpectedly occupied!`);
                    }
                    continue; // Move to next iteration, skipping normal placement for i=1
                }
                // --- END of check if this turn was a SWAP ---
            }
            // --- END SWAP RULE LOGIC Check---


            // --- Normal Move Placement Logic (for i=0 or i>1 or i=1 without swap) ---
            const player = (i % 2 === 0) ? 1 : 2; // 1 Red, 2 Blue

            if (board[y][x] === 0) {
                // Cell is empty on the board, place the stone
                board[y][x] = player;
                 //console.log(`ApplyMoves: Placed player ${player} at ${move} ([${x},${y}])`);
            } else {
                // Cell is already occupied. This indicates an issue.
                 console.warn(`ApplyMoves Warning: Attempted to play on occupied cell: ${move} ([${x}, ${y}]) by player ${player} at index ${i}. Board value: ${board[y][x]}. History: ${movesToApply.join(' ')}`);
            }
            // --- END Normal Move Placement ---
        }
    }


    // Updates the game log display
    function displayLog() {
        gameLog.innerHTML = '';
        // Show only moves up to the current index being displayed
        moves.slice(0, currentMoveIndex).forEach((move, index) => {
            const span = document.createElement('span');
            span.textContent = move + ' ';
            span.classList.add('log-move');
            span.dataset.moveIndex = index; // 0-based index

            // Highlight the move *that resulted* in the current board state
            if (index === currentMoveIndex - 1) {
                span.style.fontWeight = 'bold';
                span.style.color = 'darkblue';
            }
            gameLog.appendChild(span);
        });
    }


    // --- Victory Check Logic (DFS - unchanged) ---
    function getNeighbors(r, c, size = boardSize) { /* ... unchanged ... */
        const directions = [ [-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [1, -1] ];
        return directions
            .map(([dr, dc]) => [r + dr, c + dc])
            .filter(([nr, nc]) => nr >= 0 && nr < size && nc >= 0 && nc < size);
    }
    function dfs(startNode, player, isGoal) { /* ... unchanged ... */
        const stack = [startNode];
        const visited = new Set([`${startNode[0]},${startNode[1]}`]);
        while (stack.length > 0) {
            const [r, c] = stack.pop();
            if (isGoal(r, c)) return true;
            const neighbors = getNeighbors(r, c);
            for (const [nr, nc] of neighbors) {
                const neighborKey = `${nr},${nc}`;
                if (board[nr][nc] === player && !visited.has(neighborKey)) {
                    visited.add(neighborKey);
                    stack.push([nr, nc]);
                }
            }
        }
        return false;
     }
    function checkWinnerRed() { /* ... unchanged ... */
        const size = boardSize; const isGoal = (r, c) => r === size - 1;
        for (let c = 0; c < size; c++) {
            if (board[0][c] === 1 && dfs([0, c], 1, isGoal)) return true;
        } return false;
    }
    function checkWinnerBlue() { /* ... unchanged ... */
        const size = boardSize; const isGoal = (r, c) => c === size - 1;
        for (let r = 0; r < size; r++) {
            if (board[r][0] === 2 && dfs([r, 0], 2, isGoal)) return true;
        } return false;
    }

    function checkVictory() {
        if (isGameOver) return;
        if (checkWinnerRed()) {
            showVictoryMessage("Red wins!", "red");
        } else if (checkWinnerBlue()) {
            showVictoryMessage("Blue wins!", "blue");
        }
        // Re-check button states after potential game over
        prevMoveButton.disabled = currentMoveIndex === 0;
        nextMoveButton.disabled = currentMoveIndex === moves.length || isGameOver;
    }

    // --- UI Helper Functions (Victory Message, New Game - unchanged) ---
    function showVictoryMessage(message, color) { /* ... unchanged ... */
        if (document.querySelector('.victory-message')) return;
        isGameOver = true;

        const victoryPopup = document.createElement('div');
        victoryPopup.classList.add('victory-message');
        victoryPopup.style.backgroundColor = color === "red" ? "#F65252" : "#52BDF6";

        const messageText = document.createElement('p');
        messageText.textContent = message;
        victoryPopup.appendChild(messageText);

        const closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        closeButton.classList.add('close-button');
        closeButton.onclick = () => hideVictoryMessage(victoryPopup);
        victoryPopup.appendChild(closeButton);

        const newGameBtnPopup = document.createElement('button');
        newGameBtnPopup.textContent = 'New Game';
        newGameBtnPopup.classList.add('new-game-button');
        newGameBtnPopup.onclick = () => {
            startNewGame();
            hideVictoryMessage(victoryPopup);
        };
        victoryPopup.appendChild(newGameBtnPopup);

        document.body.appendChild(victoryPopup);
        requestAnimationFrame(() => { victoryPopup.classList.add('show'); });
     }
    function hideVictoryMessage(victoryPopup) { /* ... unchanged ... */
         if (!victoryPopup) return;
        victoryPopup.classList.remove('show');
        victoryPopup.addEventListener('transitionend', () => {
            if (victoryPopup.parentNode) victoryPopup.parentNode.removeChild(victoryPopup);
        }, { once: true });
    }

    function startNewGame() {
        const existingPopup = document.querySelector('.victory-message');
        if (existingPopup) hideVictoryMessage(existingPopup);

        isGameOver = false;
        moves = [];
        currentMoveIndex = 0;
        textInput.value = '';
        // Reset board visuals, counter, log, internal board state via displayBoardState
        displayBoardState(0);
        // Check if AI should play first
        checkAITurn();
    }


    // --- AI Integration ---
   function fetchNextMoveAI() {
        if (isGameOver) return;

        const currentMovesStr = moves.slice(0, currentMoveIndex).join(' ');
        console.log("AI thinking... Moves sent:", currentMovesStr);

        fetch('/next-move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ moves: currentMovesStr })
        })
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log("AI response:", data);
            if (data.next_moves && data.next_moves.length > 0) {
                let chosenMove = null; // The final move to add (original or swapped)

                // AI needs to pick the *best valid* move from suggestions
                for (const suggestedMove of data.next_moves) {
                    if (["<|endoftext|>", "<|startoftext|>", "<r>", "<b>", "<|pad|>", "<|unk|>"].includes(suggestedMove)) continue;

                    if (!isValidMove(suggestedMove)) {
                         console.log(`AI suggested invalid format: ${suggestedMove}`);
                         continue; // Skip invalid format
                    }

                    // --- AI SWAP Consideration ---
                    // If it's AI's (Blue's) turn (index 1) and swap is allowed
                    if (_allow_swap_rule && currentMoveIndex === 1 && moves.length === 1 && suggestedMove === moves[0]) {
                        // AI suggested swapping! Calculate swapped move.
                        const [x0, y0] = convertMoveToCoords(moves[0]);
                        const swappedMoveId = coordsToMove(y0, x0);

                        if (swappedMoveId) {
                             // For AI, we should *only* consider the swap if the target cell is valid.
                             // No need to check occupancy here as applyMoves handles it.
                             console.log("AI considers SWAP! Original:", suggestedMove, "Swapped:", swappedMoveId);
                             chosenMove = swappedMoveId; // Choose the swapped move
                             break; // Found the best move (swap)
                        } else {
                             console.warn("AI suggested swap, but swapped coords invalid?");
                             continue; // Try next suggestion
                        }
                    }
                    // --- END AI SWAP Consideration ---
                    else {
                        // Normal move suggestion: check if the cell is actually free on the internal board
                        if (!isCellOccupied(suggestedMove)) {
                            chosenMove = suggestedMove; // Found a valid, empty cell
                            break; // Use the first valid empty suggestion
                        } else {
                             console.log(`AI suggested occupied/invalid cell: ${suggestedMove}`);
                             // Continue to next suggestion
                        }
                    }
                } // End loop through AI suggestions

                if (chosenMove) {
                    console.log(`AI plays: ${chosenMove}`);
                    addMove(chosenMove); // Add the chosen move (original or swapped)
                    checkAITurn();       // Check if other AI plays next
                } else {
                    console.error('AI failed to provide a valid move from suggestions.');
                    alert("AI could not find/make a valid move.");
                    // Stop AI play?
                    // aiRedPlaysToggle.checked = false;
                    // aiBluePlaysToggle.checked = false;
                }
            } else {
                console.error('Error or no moves from AI:', data.error || 'No moves returned');
                alert(`Error getting AI move: ${data.error || 'No moves returned'}`);
            }
        })
        .catch(error => {
            console.error('Error fetching AI move:', error);
            alert(`Network error or server issue fetching AI move: ${error.message}`);
        });
    }

    function checkAITurn() {
        if (isGameOver) return;
        const isRedTurn = currentMoveIndex % 2 === 0;
        const isBlueTurn = !isRedTurn;

        if ((isRedTurn && aiRedPlaysToggle.checked) || (isBlueTurn && aiBluePlaysToggle.checked)) {
            setTimeout(fetchNextMoveAI, 300); // Delay for UX
        }
    }

    // --- Initial Setup ---
    startNewGame(); // Initialize board and state on page load

});
// --- END OF FULL UPDATED script.js ---