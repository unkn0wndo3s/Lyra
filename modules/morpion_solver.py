"""
Optimal tic-tac-toe (morpion) solver.

Given a 3x3 grid described as an array of strings `["XO ", " X ", "  O"]`,
the solver returns the best next move for X or O. Uses the standard minimax
algorithm with perfect-play heuristics:

- Win immediately if possible.
- Block opponent's immediate win.
- Prefer center, then corners, then edges.
- Detect draws to avoid unnecessary moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


Board = Tuple[Tuple[str, str, str], Tuple[str, str, str], Tuple[str, str, str]]
Move = Tuple[int, int]


def _normalize_cell(value) -> str:
    if isinstance(value, str):
        val = value.strip().upper()
        if val in {"X", "O"}:
            return val
        return " "
    return " "


def parse_board(rows: Sequence[Sequence[str] | str]) -> Board:
    if len(rows) != 3:
        raise ValueError("Board must have exactly 3 rows.")

    normalized_rows: List[Tuple[str, str, str]] = []
    for row in rows:
        if isinstance(row, str):
            if len(row) != 3:
                raise ValueError("String rows must be length 3.")
            normalized_rows.append(
                tuple(cell.upper() if cell.upper() in {"X", "O"} else " " for cell in row)  # type: ignore[arg-type]
            )
        else:
            if len(row) != 3:
                raise ValueError("Row sequences must contain 3 entries.")
            normalized_rows.append(tuple(_normalize_cell(cell) for cell in row))  # type: ignore[arg-type]

    return tuple(normalized_rows)  # type: ignore[return-value]


def board_to_rows(board: Board) -> List[str]:
    return ["".join(row) for row in board]


def current_turn(board: Board) -> str:
    flat = [cell for row in board for cell in row]
    x_count = flat.count("X")
    o_count = flat.count("O")
    return "X" if x_count == o_count else "O"


def empty_cells(board: Board) -> List[Move]:
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]


def place_move(board: Board, move: Move, player: str) -> Board:
    r, c = move
    if board[r][c] != " ":
        raise ValueError("Invalid move: cell is already occupied.")
    new_board = [list(row) for row in board]
    new_board[r][c] = player
    return tuple(tuple(row) for row in new_board)  # type: ignore[return-value]


def winner(board: Board) -> Optional[str]:
    lines = []
    lines.extend(board)
    lines.extend(zip(*board))
    lines.append(tuple(board[i][i] for i in range(3)))
    lines.append(tuple(board[i][2 - i] for i in range(3)))
    for line in lines:
        if line[0] != " " and all(cell == line[0] for cell in line):
            return line[0]
    return None


def is_draw(board: Board) -> bool:
    return winner(board) is None and all(cell != " " for row in board for cell in row)


def evaluate_board(board: Board, player: str) -> int:
    opponent = "O" if player == "X" else "X"
    win = winner(board)
    if win == player:
        return 10
    if win == opponent:
        return -10
    return 0


def minimax(board: Board, depth: int, maximizing_player: bool, player: str) -> Tuple[int, Optional[Move]]:
    current_player = player if maximizing_player else ("O" if player == "X" else "X")
    score = evaluate_board(board, player)
    if score == 10 or score == -10 or is_draw(board):
        return score - depth if score == 10 else score + depth if score == -10 else 0, None

    best_move: Optional[Move] = None
    if maximizing_player:
        best_score = -float("inf")
        for move in empty_cells(board):
            updated = place_move(board, move, current_player)
            move_score, _ = minimax(updated, depth + 1, False, player)
            if move_score > best_score:
                best_score = move_score
                best_move = move
        return best_score, best_move

    best_score = float("inf")
    for move in empty_cells(board):
        updated = place_move(board, move, current_player)
        move_score, _ = minimax(updated, depth + 1, True, player)
        if move_score < best_score:
            best_score = move_score
            best_move = move
    return best_score, best_move


def best_move(board_rows: Sequence[Sequence[str] | str], player: Optional[str] = None) -> Move:
    """
    Return the optimal move (row, col) for the given player.

    Args:
        board_rows: sequence of 3 strings (each 3 chars) describing the board.
        player: 'X' or 'O'. If None, infer from board turn.
    """

    board = parse_board(board_rows)
    if player is None:
        player = current_turn(board)
    player = player.upper()
    if player not in {"X", "O"}:
        raise ValueError("Player must be 'X' or 'O'.")

    if winner(board) or is_draw(board):
        raise ValueError("Game is already finished.")

    # Immediate win/block heuristics
    for move in empty_cells(board):
        if winner(place_move(board, move, player)) == player:
            return move

    opponent = "O" if player == "X" else "X"
    for move in empty_cells(board):
        if winner(place_move(board, move, opponent)) == opponent:
            return move

    _, move = minimax(board, 0, True, player)
    if move is None:
        raise RuntimeError("No valid moves available.")
    return move

