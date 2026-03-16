#!/usr/bin/env python3
"""
callmap_gen.py — генератор карты вызовов функций для Python-проектов.

Использование:
    python callmap_gen.py <путь_к_проекту> [-o output.md] [--exclude dir1,dir2]

Примеры:
    python callmap_gen.py ./myproject
    python callmap_gen.py ./myproject -o docs/callmap.md
    python callmap_gen.py ./myproject --exclude venv,__pycache__,.git,tests
"""

import ast
import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Структуры данных ────────────────────────────────────────────────────────

@dataclass
class CallInfo:
    """Один вызов функции внутри процедуры."""
    callee: str           # как написано в коде: "db.load_api_keys()"
    callee_module: str    # модуль, если удалось определить: "storage"
    callee_func: str      # имя функции: "load_api_keys"
    lineno: int
    context: str = ""     # краткий контекст (комментарий / имя переменной)


@dataclass
class CallerInfo:
    """Ссылка на функцию, которая вызывает текущую."""
    qualname: str    # qualname caller-функции
    rel_path: str    # файл, где живёт caller
    lineno: int      # строка вызова в caller-е


@dataclass
class ProcedureInfo:
    """Одна функция/метод в файле."""
    name: str
    qualname: str         # ClassName.method_name или просто name
    lineno: int
    calls: list[CallInfo] = field(default_factory=list)
    callers: list["CallerInfo"] = field(default_factory=list)
    is_async: bool = False
    decorator_names: list[str] = field(default_factory=list)
    docstring: str = ""


@dataclass
class FileInfo:
    """Один Python-файл."""
    path: Path
    rel_path: str
    module_name: str      # a.b.c
    imports: dict[str, str] = field(default_factory=dict)  # alias -> full_name
    procedures: list[ProcedureInfo] = field(default_factory=list)


# ─── Фильтры шума ────────────────────────────────────────────────────────────

# Встроенные функции Python — не интересны для call map
_BUILTINS = {
    "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "list", "dict", "set", "tuple", "str", "int", "float", "bool",
    "bytes", "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "delattr", "callable", "iter", "next", "any", "all", "min", "max", "sum",
    "abs", "round", "hash", "id", "repr", "format", "open", "vars", "dir",
    "super", "object", "property", "staticmethod", "classmethod",
}

# Методы, которые вызываются на произвольных объектах (dict.get, list.append, str.split…)
# Они не несут архитектурной информации
_NOISE_METHODS = {
    "get", "set", "update", "items", "keys", "values", "append", "extend",
    "insert", "remove", "pop", "clear", "copy", "count", "index", "sort",
    "reverse", "join", "split", "strip", "lstrip", "rstrip", "replace",
    "startswith", "endswith", "upper", "lower", "format", "encode", "decode",
    "read", "write", "close", "seek", "tell", "flush",
    "fetchone", "fetchall", "execute", "commit", "rollback",
    "isoformat", "strftime", "strptime", "now", "today", "fromisoformat",
    "resolve", "exists", "is_dir", "is_file", "mkdir", "rglob", "read_text",
    "write_text", "relative_to", "parent", "parts", "stem", "suffix",
    "splitlines", "removesuffix", "removeprefix",
    "classes", "style", "props",               # NiceGUI UI-шум
    "disable", "enable",
}

# Методы регистрации обработчиков событий.
# obj.on_click(handler), obj.on("event", handler), obj.connect("sig", handler) и т.д.
# Аргумент-функция из этих вызовов трактуется как вызов (логическая связь).
_EVENT_METHODS = {
    # NiceGUI / общие
    "on_click", "on_change", "on_keydown", "on_keyup", "on_keypress",
    "on_press", "on_release", "on_submit", "on_select", "on_blur", "on_focus",
    "on_upload", "on_value_change", "on_close", "on_open", "on_confirm",
    "on_cancel", "on_delete", "on_rename", "on_move", "on_drop", "on_resize",
    # Общие паттерны
    "on", "connect", "bind", "subscribe", "listen", "register",
    "add_listener", "add_handler", "set_callback", "set_handler",
    # asyncio / threading
    "add_done_callback",
    # Таймеры (callback передаётся как аргумент)
    "timer", "call_later", "call_soon", "call_at",
    "after", "schedule", "set_interval", "set_timeout",
}

# Модули стандартной библиотеки и популярных lib — их вызовы менее интересны
# но мы не фильтруем полностью, только помечаем
_STDLIB_MODULES = {
    "os", "sys", "re", "json", "time", "math", "random", "copy", "io",
    "pathlib", "logging", "threading", "hashlib", "hmac", "datetime",
    "collections", "itertools", "functools", "contextlib", "dataclasses",
    "typing", "abc", "enum", "struct", "traceback", "inspect",
    "requests", "sqlite3", "csv", "ast", "argparse", "shutil", "glob",
}

# Методы-исполнители: первый аргумент — callable, остальные — его параметры.
# run.io_bound(fn, a, b) → fn является вызовом.
_EXECUTOR_METHODS = {
    "io_bound", "cpu_bound",                      # nicegui.run
    "create_task", "ensure_future",               # asyncio
    "run_coroutine_threadsafe", "run_in_executor",
    "submit", "map",                              # concurrent.futures / multiprocessing
    "apply", "apply_async", "starmap", "starmap_async",
    "spawn", "dispatch", "defer", "delay",        # общие паттерны
    "enqueue", "schedule_call",
}

# Пользовательский список исключённых библиотек/модулей.
# Заполняется из --exclude-libs перед парсингом проекта.
_EXCLUDED_LIBS: set[str] = set()


def _is_noise(call: "CallInfo") -> bool:
    """True если вызов — шум (builtin, метод на переменной, исключённая библиотека)."""
    # Встроенные функции
    if call.callee_func in _BUILTINS:
        return True
    # Вызов на локальной переменной: ticker.get(), row.items(), response.json()
    if call.callee_module == _VAR_:
        return True
    # Исключённые пользователем библиотеки: nicegui, datetime, requests, ...
    # Проверяем и точное совпадение, и префикс (nicegui.ui → nicegui)
    if _EXCLUDED_LIBS:
        mod_root = call.callee_module.split(".")[0] if call.callee_module else ""
        if call.callee_module in _EXCLUDED_LIBS or mod_root in _EXCLUDED_LIBS:
            return True
    return False


# ─── Сбор импортов ───────────────────────────────────────────────────────────

def collect_imports(tree: ast.Module) -> dict[str, str]:
    """
    Возвращает словарь alias -> полное имя.
    import storage as db  →  {"db": "storage"}
    from bybit_api import sync_transactions_from_bybit  →  {"sync_transactions_from_bybit": "bybit_api.sync_transactions_from_bybit"}
    from tabs import tab_portfolio  →  {"tab_portfolio": "tabs.tab_portfolio"}
    """
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name.split(".")[0]
                imports[local_name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                full = f"{module}.{alias.name}" if module else alias.name
                imports[local_name] = full
    return imports


# ─── Извлечение вызовов из AST-узла ─────────────────────────────────────────

_VAR_ = "_VAR_"  # маркер: объект — локальная переменная, не модуль из импортов


def _resolve_call(node: ast.Call, imports: dict[str, str]) -> Optional[CallInfo]:
    """Превратить ast.Call в CallInfo.

    Ключевое правило: callee_module выставляется в _VAR_ если объект
    не найден в imports — значит это локальная переменная (ticker, row, data…),
    а не импортированный модуль/пакет.
    """
    func = node.func

    if isinstance(func, ast.Name):
        name = func.id
        if name in imports:
            full = imports[name]
            parts = full.rsplit(".", 1)
            mod = parts[0] if len(parts) == 2 else ""
            fn  = parts[1] if len(parts) == 2 else full
        else:
            # Простой вызов функции без модуля: calculate_fifo_profit(...)
            mod = ""
            fn  = name
        return CallInfo(
            callee=f"{name}()",
            callee_module=mod,
            callee_func=fn,
            lineno=node.lineno,
        )

    if isinstance(func, ast.Attribute):
        attr = func.attr
        # obj.method()
        if isinstance(func.value, ast.Name):
            obj = func.value.id
            if obj in imports:
                # db.load_api_keys() — obj это импортированный модуль
                mod = imports[obj]
            else:
                # ticker.get(), row.items(), response.json() — obj это переменная
                mod = _VAR_
            callee_str = f"{obj}.{attr}()"
            return CallInfo(
                callee=callee_str,
                callee_module=mod,
                callee_func=attr,
                lineno=node.lineno,
            )
        # obj.sub.method()
        if isinstance(func.value, ast.Attribute):
            parts = []
            cur = func.value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            root_name = None
            if isinstance(cur, ast.Name):
                root_name = cur.id
                parts.append(root_name)
                parts.reverse()
                full_chain = ".".join(parts)
                # Если корень цепочки не в импортах — тоже переменная
                if root_name not in imports:
                    return CallInfo(
                        callee=f"{full_chain}.{attr}()",
                        callee_module=_VAR_,
                        callee_func=attr,
                        lineno=node.lineno,
                    )
                # Раскрываем алиас корня: ui.navigate → nicegui.ui.navigate
                resolved_root = imports[root_name]
                middle = ".".join(parts[1:])  # всё между root и attr
                full_chain = f"{resolved_root}.{middle}" if middle else resolved_root
            else:
                parts.reverse()
                full_chain = ".".join(parts)
            callee_str = f"{full_chain}.{attr}()"
            return CallInfo(
                callee=callee_str,
                callee_module=full_chain,
                callee_func=attr,
                lineno=node.lineno,
            )

    return None


def _add_callback(cb_name, lineno, imports, calls, seen):
    """Добавить функцию-callback как вызов."""
    if cb_name in _BUILTINS:
        return
    if cb_name in imports:
        full = imports[cb_name]
        parts = full.rsplit(".", 1)
        cb_mod = parts[0] if len(parts) == 2 else ""
        cb_fn  = parts[1] if len(parts) == 2 else full
    else:
        cb_mod = ""
        cb_fn  = cb_name
    cb_info = CallInfo(callee=f"{cb_name}()", callee_module=cb_mod,
                       callee_func=cb_fn, lineno=lineno)
    key = (cb_info.callee, cb_info.lineno)
    if key not in seen:
        seen.add(key)
        calls.append(cb_info)


def _walk_no_inner_funcs(nodes):
    """Обходит дерево операторов, пропуская вложенные FunctionDef/AsyncFunctionDef целиком.
    Вызовы из closure приписываются только самой closure, а не родительской функции.
    """
    queue = list(nodes)
    while queue:
        node = queue.pop()
        # Вложенные функции пропускаем полностью — не yield и не recurse
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield node
        for child in ast.iter_child_nodes(node):
            queue.append(child)


def extract_calls(body_nodes: list[ast.stmt], imports: dict[str, str]) -> list[CallInfo]:
    """Обойти тело функции и собрать все вызовы (не заходя в тела вложенных функций)."""
    calls: list[CallInfo] = []
    seen: set[tuple] = set()
    cb_calls: list[CallInfo] = []   # callback-связи (не фильтруются шумом)
    cb_seen: set[tuple] = set()

    for node in _walk_no_inner_funcs(body_nodes):
        if not isinstance(node, ast.Call):
            continue

        # ── Обычный вызов ──────────────────────────────────────────────────────
        info = _resolve_call(node, imports)
        if info:
            key = (info.callee, info.lineno)
            if key not in seen:
                seen.add(key)
                calls.append(info)

        # ── Паттерн регистрации обработчика: obj.on_click(handler) ─────────────
        # Любой аргумент-Name из _EVENT_METHODS является callback.
        if isinstance(node.func, ast.Attribute) and node.func.attr in _EVENT_METHODS:
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Name) and arg.id not in _BUILTINS:
                    _add_callback(arg.id, node.lineno, imports, cb_calls, cb_seen)

        # ── Паттерн executor: run.io_bound(fn, arg1, arg2) ──────────────────
        # Первый аргумент — callable, остальные — его параметры.
        # Добавляем в cb_calls независимо от _is_noise на самом вызове.
        if isinstance(node.func, ast.Attribute) and node.func.attr in _EXECUTOR_METHODS:
            if node.args:
                first = node.args[0]
                if isinstance(first, ast.Name):
                    _add_callback(first.id, node.lineno, imports, cb_calls, cb_seen)
                elif isinstance(first, ast.Call) and isinstance(first.func, ast.Name):
                    _add_callback(first.func.id, node.lineno, imports, cb_calls, cb_seen)

    # Шумовой фильтр только для обычных вызовов; callback-связи всегда валидны
    calls = [c for c in calls if not _is_noise(c)]
    normal_keys = {(c.callee, c.lineno) for c in calls}
    for c in cb_calls:
        if (c.callee, c.lineno) not in normal_keys:
            calls.append(c)
    return sorted(calls, key=lambda c: c.lineno)


# ─── Обход файла ─────────────────────────────────────────────────────────────

def _decorator_names(decorator_list: list[ast.expr]) -> list[str]:
    names = []
    for d in decorator_list:
        if isinstance(d, ast.Name):
            names.append(d.id)
        elif isinstance(d, ast.Attribute):
            names.append(f"{ast.unparse(d)}")
        elif isinstance(d, ast.Call):
            names.append(f"{ast.unparse(d.func)}(...)")
    return names


def parse_file(path: Path, rel_path: str, module_name: str) -> Optional[FileInfo]:
    """Разобрать один Python-файл, вернуть FileInfo."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        print(f"  [WARN] SyntaxError в {rel_path}: {e}", file=sys.stderr)
        return None

    imports = collect_imports(tree)
    file_info = FileInfo(path=path, rel_path=rel_path, module_name=module_name, imports=imports)

    # Обходим верхний уровень и классы
    def _visit_body(nodes: list[ast.stmt], class_name: str = ""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{class_name}.{node.name}" if class_name else node.name
                proc = ProcedureInfo(
                    name=node.name,
                    qualname=qualname,
                    lineno=node.lineno,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    decorator_names=_decorator_names(node.decorator_list),
                    docstring=ast.get_docstring(node) or "",
                )
                proc.calls = extract_calls(node.body, imports)
                file_info.procedures.append(proc)

                # Вложенные функции — тоже собираем как отдельные
                for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not node:
                        inner_qualname = f"{qualname}.<locals>.{child.name}"
                        inner_proc = ProcedureInfo(
                            name=child.name,
                            qualname=inner_qualname,
                            lineno=child.lineno,
                            is_async=isinstance(child, ast.AsyncFunctionDef),
                            decorator_names=_decorator_names(child.decorator_list),
                            docstring=ast.get_docstring(child) or "",
                        )
                        inner_proc.calls = extract_calls(child.body, imports)
                        if inner_proc.calls:  # только если есть вызовы
                            file_info.procedures.append(inner_proc)

            elif isinstance(node, ast.ClassDef):
                _visit_body(node.body, class_name=node.name)

    _visit_body(tree.body)

    # ── Код верхнего уровня (<module>) ────────────────────────────────────────
    # Собираем вызовы из операторов верхнего уровня, которые НЕ являются
    # определениями функций/классов (они уже обработаны выше).
    # Это покрывает: if __name__ == "__main__", голые вызовы, присваивания.
    top_level_stmts = [
        node for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Import, ast.ImportFrom))
    ]
    if top_level_stmts:
        module_calls = extract_calls(top_level_stmts, imports)
        if module_calls:
            module_proc = ProcedureInfo(
                name="<module>",
                qualname="<module>",
                lineno=1,
                docstring="Код верхнего уровня модуля (вне функций и классов).",
            )
            module_proc.calls = module_calls
            file_info.procedures.append(module_proc)

    return file_info


# ─── Сканирование проекта ─────────────────────────────────────────────────────

DEFAULT_EXCLUDE = {
    "__pycache__", ".git", ".hg", ".svn", "venv", ".venv",
    "env", ".env", "node_modules", "dist", "build", ".tox",
    "migrations", ".mypy_cache", ".pytest_cache",
}


def scan_project(root: Path, extra_exclude: set[str] = frozenset()) -> list[FileInfo]:
    exclude = DEFAULT_EXCLUDE | extra_exclude
    files: list[FileInfo] = []

    for py_file in sorted(root.rglob("*.py")):
        # Фильтр по исключённым папкам
        parts = py_file.relative_to(root).parts
        if any(p in exclude for p in parts):
            continue

        rel_path = str(py_file.relative_to(root))
        # Модульное имя: app/tabs/tab_portfolio.py → app.tabs.tab_portfolio
        module_name = rel_path.replace(os.sep, ".").removesuffix(".py")

        info = parse_file(py_file, rel_path, module_name)
        if info:
            files.append(info)

    return files


# ─── Разрешение вызовов по всему проекту ─────────────────────────────────────

def _build_module_index(files: list[FileInfo]) -> dict[str, str]:
    """
    Строит индекс: последний сегмент модуля → полное имя.
    tabs.tab_portfolio → tabs/tab_portfolio.py (rel_path)
    """
    index: dict[str, str] = {}
    for f in files:
        parts = f.module_name.split(".")
        for i in range(len(parts)):
            key = ".".join(parts[i:])
            if key not in index:
                index[key] = f.rel_path
    return index


def build_callers_index(files: list[FileInfo], module_index: dict[str, str]) -> None:
    """
    Заполняет поле .callers у каждой ProcedureInfo.

    Для каждой процедуры P в файле F, для каждого вызова C:
      1. Определяем целевой файл по callee_module
      2. Находим процедуру с именем callee_func
      3. Добавляем в её .callers ссылку на P
    """
    # Индекс: (rel_path, func_name) -> ProcedureInfo
    proc_index: dict[tuple, ProcedureInfo] = {}
    for f in files:
        for proc in f.procedures:
            proc_index[(f.rel_path, proc.name)]     = proc
            proc_index[(f.rel_path, proc.qualname)] = proc

    for caller_file in files:
        for caller_proc in caller_file.procedures:
            for call in caller_proc.calls:
                if not call.callee_module or call.callee_module == _VAR_:
                    target_rel = caller_file.rel_path
                else:
                    target_rel = resolve_module_to_file(call.callee_module, module_index)
                    if not target_rel:
                        continue

                callee_proc = (
                    proc_index.get((target_rel, call.callee_func)) or
                    proc_index.get((target_rel, call.callee_func.split(".")[-1]))
                )
                if callee_proc is None:
                    continue

                # Не добавляем дубликаты
                already = any(
                    c.qualname == caller_proc.qualname and c.rel_path == caller_file.rel_path
                    for c in callee_proc.callers
                )
                if not already:
                    callee_proc.callers.append(CallerInfo(
                        qualname=caller_proc.qualname,
                        rel_path=caller_file.rel_path,
                        lineno=call.lineno,
                    ))


def orphan_files(files: list[FileInfo]) -> list[FileInfo]:
    """Файлы, ни одна процедура которых не вызывается из другого файла проекта."""
    result = []
    for f in files:
        has_external_caller = any(
            any(c.rel_path != f.rel_path for c in proc.callers)
            for proc in f.procedures
        )
        if not has_external_caller:
            result.append(f)
    return result


def resolve_module_to_file(module: str, module_index: dict[str, str]) -> Optional[str]:
    """Попробовать найти файл по имени модуля или его суффиксу."""
    if module in module_index:
        return module_index[module]
    # Попробуем по последнему сегменту
    last = module.split(".")[-1]
    return module_index.get(last)


# ─── Рендер Markdown ─────────────────────────────────────────────────────────

def _anchor(text: str) -> str:
    """GitHub-совместимый якорь."""
    return text.lower().replace(" ", "-").replace("/", "").replace(".", "").replace("_", "_").replace("(", "").replace(")", "")


def render_markdown(files: list[FileInfo], module_index: dict[str, str], project_name: str) -> str:
    lines: list[str] = []

    lines.append(f"# Карта вызовов функций — {project_name}\n")
    lines.append("> Автоматически сгенерировано `callmap_gen.py`\n")
    lines.append(f"> Файлов: **{len(files)}** | Процедур: **{sum(len(f.procedures) for f in files)}**\n")

    # Оглавление
    lines.append("## Содержание\n")
    for f in files:
        if not f.procedures:
            continue
        anchor = _anchor(f"file-{f.rel_path}")
        lines.append(f"- [{f.rel_path}](#{anchor})")
    lines.append("")

    lines.append("---\n")

    # ── Модули-сироты ──────────────────────────────────────────────────────────
    orphans = orphan_files(files)
    if orphans:
        lines.append("## ⚠️ Модули без входящих вызовов\n")
        lines.append("> Файлы, ни одна функция которых не вызывается из других модулей проекта.\n")
        for f in orphans:
            anchor = _anchor(f"file-{f.rel_path}")
            n_procs = len(f.procedures)
            lines.append(f"- [`{f.rel_path}`](#{anchor}) — {n_procs} процедур")
        lines.append("")
        lines.append("---\n")

    # Тело
    for f in files:
        if not f.procedures:
            continue

        anchor = _anchor(f"file-{f.rel_path}")
        lines.append(f'<a name="{anchor}"></a>')
        lines.append(f"## `{f.rel_path}`\n")

        # Импорты (коротко)
        if f.imports:
            alias_lines = []
            for alias, full in sorted(f.imports.items()):
                if alias != full.split(".")[-1]:
                    alias_lines.append(f"`{alias}` → `{full}`")
                else:
                    alias_lines.append(f"`{full}`")
            lines.append("**Импорты:** " + ", ".join(alias_lines) + "\n")

        procs = sorted(f.procedures, key=lambda p: p.lineno)
        for proc in procs:
            # Заголовок процедуры
            prefix = "async " if proc.is_async else ""
            deco = ""
            if proc.decorator_names:
                deco = " — `@" + "`, `@".join(proc.decorator_names) + "`"
            lines.append(f"### `{prefix}{proc.qualname}()`{deco}\n")
            lines.append(f"*строка {proc.lineno}*\n")

            if proc.docstring:
                doc_lines = proc.docstring.strip().splitlines()
                summary = doc_lines[0].strip()
                rest = "\n".join(doc_lines[1:]).strip()
                if rest:
                    lines.append(f"> {summary}\n")
                    lines.append(f"> <details><summary>подробнее</summary>\n>\n> ```\n> {rest}\n> ```\n> </details>\n")
                else:
                    lines.append(f"> {summary}\n")

            if not proc.calls:
                lines.append("*Внешних вызовов не обнаружено.*\n")
            else:
                # Группируем по модулю
                by_module: dict[str, list[CallInfo]] = defaultdict(list)
                local_calls: list[CallInfo] = []

                for call in proc.calls:
                    if call.callee_module:
                        by_module[call.callee_module].append(call)
                    else:
                        local_calls.append(call)

                if local_calls:
                    by_module["*(локальные)*"] = local_calls

                for mod, mod_calls in sorted(by_module.items()):
                    resolved = resolve_module_to_file(mod, module_index)
                    if resolved and resolved != f.rel_path:
                        mod_display = f"[`{mod}`](#{_anchor(f'file-{resolved}')})"
                    else:
                        mod_display = f"`{mod}`"

                    lines.append(f"**{mod_display}**")
                    for call in mod_calls:
                        lines.append(f"- `{call.callee}` *(строка {call.lineno})*")
                    lines.append("")

            # Callers
            if proc.callers:
                lines.append("**Вызывается из:**")
                from collections import Counter
                ln_count: dict = Counter((c.rel_path, c.lineno) for c in proc.callers)
                ln_shown: set = set()
                for caller in sorted(proc.callers, key=lambda c: (c.rel_path, c.lineno, c.qualname)):
                    caller_anchor = _anchor(f"file-{caller.rel_path}")
                    key = (caller.rel_path, caller.lineno)
                    show_ln = key not in ln_shown
                    if ln_count[key] > 1:
                        ln_shown.add(key)
                    lineno_str = f" *(строка {caller.lineno})*" if show_ln else ""
                    lines.append(
                        f"- [`{caller.qualname}()`](#{caller_anchor})"
                        f" — `{caller.rel_path}`{lineno_str}"
                    )
                lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


# ─── Рендер HTML ─────────────────────────────────────────────────────────────

def _he(text: str) -> str:
    """Минимальное HTML-экранирование."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_html(files: list[FileInfo], module_index: dict[str, str], project_name: str) -> str:
    from datetime import datetime

    total_procs = sum(len(f.procedures) for f in files)
    total_calls = sum(len(p.calls) for f in files for p in f.procedures)
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")

    # ── Sidebar items ──────────────────────────────────────────────────────────
    sidebar_items: list[str] = []
    for f in files:
        fid = f"file-{f.rel_path.replace(os.sep, '-').replace('.', '-')}"
        procs_sorted = sorted(f.procedures, key=lambda p: p.lineno)
        # Список процедур внутри details
        proc_links = ""
        for proc in procs_sorted:
            pid = f"{fid}-{proc.qualname.replace('.', '-').replace('<', '').replace('>', '')}"
            prefix = "async " if proc.is_async else ""
            proc_links += (
                f'<a class="sidebar-proc" href="#{pid}" '
                f'title="{_he(prefix + proc.qualname + "()")}">'
                f'{_he(proc.qualname)}()</a>'
            )
        sidebar_items.append(
            f'<details class="sidebar-group" data-fid="{fid}">'
            f'<summary class="sidebar-file" title="{_he(f.rel_path)}">'
            f'<span class="sidebar-icon">📄</span>'
            f'<span class="sidebar-fname">{_he(f.rel_path)}</span>'
            f'<span class="sidebar-count">{len(procs_sorted)}</span>'
            f'</summary>'
            f'<div class="sidebar-procs">{proc_links}</div>'
            f'</details>'
        )

    # ── Main content ───────────────────────────────────────────────────────────
    content_blocks: list[str] = []

    for f in files:
        fid = f"file-{f.rel_path.replace(os.sep, '-').replace('.', '-')}"
        procs = sorted(f.procedures, key=lambda p: p.lineno)

        # Импорты
        import_tags: list[str] = []
        for alias, full in sorted(f.imports.items()):
            if alias != full.split(".")[-1]:
                import_tags.append(f'<span class="import-tag"><span class="import-alias">{_he(alias)}</span> → {_he(full)}</span>')
            else:
                import_tags.append(f'<span class="import-tag">{_he(full)}</span>')

        imports_html = ""
        if import_tags:
            imports_html = f'<div class="imports-row"><span class="imports-label">imports</span>{"".join(import_tags)}</div>'

        # Процедуры
        proc_blocks: list[str] = []
        for proc in procs:
            pid = f"{fid}-{proc.qualname.replace('.', '-').replace('<', '').replace('>', '')}"
            prefix = '<span class="kw-async">async</span> ' if proc.is_async else ''
            decos_html = ""
            if proc.decorator_names:
                decos_html = "".join(
                    f'<span class="decorator">@{_he(d)}</span>' for d in proc.decorator_names
                )

            # Сгруппировать вызовы по модулю
            by_module: dict[str, list[CallInfo]] = defaultdict(list)
            local_calls: list[CallInfo] = []
            for call in proc.calls:
                if call.callee_module:
                    by_module[call.callee_module].append(call)
                else:
                    local_calls.append(call)
            if local_calls:
                by_module["(локальные)"] = local_calls

            # Docstring
            if proc.docstring:
                doc_lines = proc.docstring.strip().splitlines()
                summary = doc_lines[0].strip()
                rest = "\n".join(l.strip() for l in doc_lines[1:]).strip()
                if rest:
                    doc_html = (
                        f'<div class="proc-doc">' +
                        f'<span class="doc-summary">{_he(summary)}</span>' +
                        f'<details class="doc-details"><summary>подробнее</summary>' +
                        f'<pre class="doc-rest">{_he(rest)}</pre></details>' +
                        f'</div>'
                    )
                else:
                    doc_html = f'<div class="proc-doc"><span class="doc-summary">{_he(summary)}</span></div>'
            else:
                doc_html = ""

            if not proc.calls:
                calls_html = '<div class="no-calls">нет внешних вызовов</div>'
            else:
                group_blocks: list[str] = []
                for mod, mod_calls in sorted(by_module.items()):
                    resolved = resolve_module_to_file(mod, module_index)
                    mod_fid = f"file-{resolved.replace(os.sep, '-').replace('.', '-')}" if resolved else None

                    if mod_fid and resolved != f.rel_path:
                        mod_label = f'<a class="mod-link" href="#{mod_fid}">{_he(mod)}</a>'
                    else:
                        mod_label = f'<span class="mod-local">{_he(mod)}</span>'

                    call_items = "".join(
                        f'<li><code class="call-name">{_he(call.callee)}</code>'
                        f'<span class="lineno">:{call.lineno}</span></li>'
                        for call in mod_calls
                    )
                    group_blocks.append(
                        f'<div class="call-group">'
                        f'<div class="call-group-header">{mod_label}</div>'
                        f'<ul class="call-list">{call_items}</ul>'
                        f'</div>'
                    )
                calls_html = "".join(group_blocks)

            # Callers HTML
            if proc.callers:
                caller_items = ""
                # Считаем сколько раз встречается каждый (rel_path, lineno) —
                # если несколько caller'ов из одного места (напр. render + вложенная),
                # показываем lineno только у первого чтобы избежать дубликатов.
                from collections import Counter
                lineno_count: dict = Counter((c.rel_path, c.lineno) for c in proc.callers)
                lineno_shown: set = set()
                for c in sorted(proc.callers, key=lambda c: (c.rel_path, c.lineno, c.qualname)):
                    cfid = f"file-{c.rel_path.replace(os.sep, '-').replace('.', '-')}"
                    cpid = f"{cfid}-{c.qualname.replace('.', '-').replace('<', '').replace('>', '')}"
                    key = (c.rel_path, c.lineno)
                    show_lineno = key not in lineno_shown
                    if lineno_count[key] > 1:
                        lineno_shown.add(key)
                    lineno_html = f'<span class="lineno">:{c.lineno}</span>' if show_lineno else ''
                    caller_items += (
                        f'<li>'
                        f'<a class="caller-link" href="#{cpid}"><code>{_he(c.qualname)}()</code></a>'
                        f'<span class="caller-file">{_he(c.rel_path)}</span>'
                        f'{lineno_html}'
                        f'</li>'
                    )
                callers_html = (
                    f'<div class="callers-section">'
                    f'<div class="callers-label">⬆ вызывается из</div>'
                    f'<ul class="callers-list">{caller_items}</ul>'
                    f'</div>'
                )
            else:
                callers_html = ""

            proc_blocks.append(
                f'<div class="proc-card" id="{pid}">'
                f'  <div class="proc-header">'
                f'    <div class="proc-title">{decos_html}{prefix}<span class="proc-name">{_he(proc.qualname)}</span><span class="proc-parens">()</span></div>'
                f'    <span class="proc-line">L{proc.lineno}</span>'
                f'  </div>'
                f'  {doc_html}'
                f'  <div class="proc-body">{calls_html}</div>'
                f'  {callers_html}'
                f'</div>'
            )

        content_blocks.append(
            f'<section class="file-section" id="{fid}">'
            f'  <div class="file-header">'
            f'    <span class="file-icon">📄</span>'
            f'    <h2 class="file-title">{_he(f.rel_path)}</h2>'
            f'    <span class="file-stats">{len(procs)} процедур</span>'
            f'  </div>'
            f'  {imports_html}'
            f'  <div class="procs-grid">{"".join(proc_blocks)}</div>'
            f'</section>'
        )

    sidebar_html = "\n".join(sidebar_items)

    # ── Виджет «Модули без входящих вызовов» ──────────────────────────────────
    orphans = orphan_files(files)
    if orphans:
        rows = []
        for f in orphans:
            fid = f"file-{f.rel_path.replace(os.sep, '-').replace('.', '-')}"
            n = len(f.procedures)
            n_calls_out = sum(len(p.calls) for p in f.procedures)
            rows.append(
                f'<tr class="orp-row" onclick="location.hash=\'#{fid}\';" style="cursor:pointer">' +
                f'<td class="orp-file"><a href="#{fid}" onclick="event.stopPropagation()">{_he(f.rel_path)}</a></td>' +
                f'<td class="orp-n">{n}</td>' +
                f'<td class="orp-n">{n_calls_out}</td>' +
                f'</tr>'
            )
        orphan_widget = (
            '<section class="orp-section" id="orphan-modules">' +
            '<div class="orp-header">' +
            '  <span class="orp-icon">⚠️</span>' +
            f'  <h2 class="orp-title">Модули без входящих вызовов</h2>' +
            f'  <span class="orp-badge">{len(orphans)}</span>' +
            '</div>' +
            '<p class="orp-desc">Ни одна функция этих файлов не вызывается из других модулей проекта. ' +
            'Возможно, это точки входа, устаревший код или изолированные утилиты.</p>' +
            '<table class="orp-table">' +
            '<thead><tr><th>Файл</th><th title="Процедур">Проц.</th><th title="Исходящих вызовов">Вызовов</th></tr></thead>' +
            f'<tbody>{"".join(rows)}</tbody>' +
            '</table></section>'
        )
    else:
        orphan_widget = ""

    content_html = orphan_widget + ("\n" if orphan_widget else "") + "\n".join(content_blocks)

    orphan_sidebar_link = (
        '<a class="sidebar-orphan-link" href="#orphan-modules">' +
        f'⚠️ Модули-сироты <span class="sidebar-count">{len(orphans)}</span></a>'
        if orphans else ""
    )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Call Map — {_he(project_name)}</title>
<style>
  /* ── Reset & base ── */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:        #0d1117;
    --bg2:       #161b22;
    --bg3:       #1c2128;
    --border:    #30363d;
    --border2:   #484f58;
    --text:      #e6edf3;
    --text2:     #8b949e;
    --text3:     #6e7681;
    --accent:    #58a6ff;
    --accent2:   #388bfd;
    --green:     #3fb950;
    --yellow:    #d29922;
    --purple:    #bc8cff;
    --red:       #f85149;
    --font-mono: "JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace;
    --font-ui:   -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --sidebar-w: 260px;
    --radius:    6px;
  }}
  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-ui);
    font-size: 13px;
    line-height: 1.5;
    display: flex;
    min-height: 100vh;
  }}

  /* ── Sidebar ── */
  #sidebar {{
    width: var(--sidebar-w);
    min-width: var(--sidebar-w);
    background: var(--bg2);
    border-right: 1px solid var(--border);
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }}
  .sidebar-head {{
    padding: 16px 14px 12px;
    border-bottom: 1px solid var(--border);
  }}
  .sidebar-project {{
    font-size: 14px;
    font-weight: 700;
    color: var(--accent);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .sidebar-meta {{
    font-size: 11px;
    color: var(--text3);
    margin-top: 3px;
  }}
  .sidebar-search {{
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
  }}
  .sidebar-search input {{
    width: 100%;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    font-size: 12px;
    padding: 5px 8px;
    outline: none;
    font-family: var(--font-ui);
  }}
  .sidebar-search input:focus {{ border-color: var(--accent2); }}
  .sidebar-search input::placeholder {{ color: var(--text3); }}
  .sidebar-files {{
    padding: 6px 0;
    flex: 1;
    overflow-y: auto;
  }}
  .sidebar-group {{
    border-left: 2px solid transparent;
    transition: border-color 0.1s;
  }}
  .sidebar-group.active-group {{ border-left-color: var(--accent); }}
  .sidebar-group[open] > summary .sidebar-icon {{ content: ""; }}
  .sidebar-file {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px 5px 12px;
    color: var(--text2);
    font-size: 12px;
    cursor: pointer;
    list-style: none;
    transition: background 0.1s, color 0.1s;
    user-select: none;
  }}
  .sidebar-file::-webkit-details-marker {{ display: none; }}
  .sidebar-file:hover {{ background: var(--bg3); color: var(--text); }}
  .sidebar-group[open] > .sidebar-file {{ color: var(--text); }}
  .sidebar-group.active-group > .sidebar-file {{ color: var(--accent); }}
  .sidebar-fname {{
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .sidebar-count {{
    flex-shrink: 0;
    font-size: 10px;
    color: var(--text3);
    background: var(--bg3);
    border-radius: 8px;
    padding: 1px 5px;
    margin-left: auto;
  }}
  .sidebar-procs {{
    padding: 2px 0 4px 0;
  }}
  .sidebar-proc {{
    display: block;
    padding: 3px 10px 3px 28px;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text3);
    text-decoration: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border-left: 2px solid transparent;
    transition: background 0.1s, color 0.1s;
    margin-left: 12px;
  }}
  .sidebar-proc:hover {{ background: var(--bg3); color: var(--text2); }}
  .sidebar-proc.active {{ color: var(--accent); border-left-color: var(--accent); background: var(--bg3); }}
  .sidebar-icon {{ flex-shrink: 0; font-size: 11px; }}
  .sidebar-orphan-link {{
    display: flex; align-items: center; gap: 6px;
    padding: 6px 14px; margin-bottom: 4px;
    color: var(--yellow); font-size: 12px; text-decoration: none;
    background: rgba(210,153,34,.07);
    border-left: 2px solid var(--yellow);
    transition: background .1s;
  }}
  .sidebar-orphan-link:hover {{ background: rgba(210,153,34,.14); }}
  .sidebar-gen {{
    padding: 8px 14px;
    font-size: 10px;
    color: var(--text3);
    border-top: 1px solid var(--border);
  }}

  /* ── Orphan modules widget ── */
  .orp-section {{
    background: var(--bg2);
    border: 1px solid var(--yellow);
    border-left: 4px solid var(--yellow);
    border-radius: var(--radius);
    padding: 18px 22px 14px;
    margin-bottom: 28px;
  }}
  .orp-header {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
  }}
  .orp-title {{
    font-size: 15px; font-weight: 700; color: var(--yellow); flex: 1; margin: 0;
  }}
  .orp-badge {{
    background: rgba(210,153,34,.15); color: var(--yellow);
    border: 1px solid rgba(210,153,34,.4);
    border-radius: 10px; padding: 1px 9px; font-size: 12px; font-weight: 600;
  }}
  .orp-desc {{
    font-size: 12px; color: var(--text2); margin: 0 0 12px; line-height: 1.5;
  }}
  .orp-table {{
    width: 100%; border-collapse: collapse; font-size: 12px;
  }}
  .orp-table th {{
    text-align: left; padding: 5px 10px 5px 0;
    border-bottom: 1px solid var(--border2);
    color: var(--text3); font-size: 11px;
    text-transform: uppercase; letter-spacing: .5px; font-weight: 600;
  }}
  .orp-table td {{ padding: 5px 10px 5px 0; border-bottom: 1px solid var(--border); }}
  .orp-row:last-child td {{ border-bottom: none; }}
  .orp-row:hover {{ background: var(--bg3); }}
  .orp-file a {{
    color: var(--accent); text-decoration: none;
    font-family: var(--font-mono); font-size: 12px;
  }}
  .orp-file a:hover {{ text-decoration: underline; }}
  .orp-n {{ color: var(--text3); font-family: var(--font-mono); text-align: right; padding-right: 20px; }}

  /* ── Main content ── */
  #main {{
    flex: 1;
    min-width: 0;
    padding: 24px 28px;
    max-width: 1100px;
  }}
  .page-title {{
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
  }}
  .page-meta {{
    font-size: 12px;
    color: var(--text3);
    margin-bottom: 24px;
    display: flex;
    gap: 16px;
  }}
  .page-meta span {{ display: flex; align-items: center; gap: 4px; }}

  /* ── Search bar ── */
  .main-search {{
    margin-bottom: 20px;
  }}
  .main-search input {{
    width: 100%;
    max-width: 480px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    font-size: 13px;
    padding: 7px 12px;
    outline: none;
    font-family: var(--font-ui);
  }}
  .main-search input:focus {{ border-color: var(--accent2); box-shadow: 0 0 0 2px rgba(56,139,253,0.15); }}
  .main-search input::placeholder {{ color: var(--text3); }}

  /* ── File section ── */
  .file-section {{
    margin-bottom: 36px;
    scroll-margin-top: 16px;
  }}
  .file-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) 0 0;
    border-bottom: 1px solid var(--border2);
  }}
  .file-icon {{ font-size: 14px; flex-shrink: 0; }}
  .file-title {{
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 600;
    color: var(--accent);
    flex: 1;
  }}
  .file-stats {{
    font-size: 11px;
    color: var(--text3);
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1px 8px;
  }}
  .imports-row {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-top: none;
    padding: 7px 14px;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    align-items: center;
  }}
  .imports-label {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text3);
    margin-right: 4px;
  }}
  .import-tag {{
    font-family: var(--font-mono);
    font-size: 11px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 6px;
    color: var(--text2);
    white-space: nowrap;
  }}
  .import-alias {{ color: var(--yellow); }}

  /* ── Procs grid ── */
  .procs-grid {{
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 var(--radius) var(--radius);
    overflow: hidden;
  }}
  .proc-card {{
    border-bottom: 1px solid var(--border);
    scroll-margin-top: 16px;
    transition: background 0.1s;
  }}
  .proc-card:last-child {{ border-bottom: none; }}
  .proc-card:hover {{ background: rgba(56,139,253,0.03); }}
  .proc-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    background: var(--bg2);
    cursor: pointer;
    user-select: none;
    gap: 10px;
  }}
  .proc-header:hover {{ background: var(--bg3); }}
  .proc-title {{
    font-family: var(--font-mono);
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
  }}
  .kw-async {{ color: var(--purple); font-style: italic; }}
  .proc-name {{ color: var(--green); font-weight: 600; }}
  .proc-parens {{ color: var(--text3); }}
  .decorator {{
    font-size: 11px;
    color: var(--yellow);
    background: rgba(210,153,34,0.1);
    border-radius: 3px;
    padding: 0 4px;
    margin-right: 4px;
  }}
  .proc-line {{
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text3);
    flex-shrink: 0;
  }}
  .proc-body {{
    padding: 10px 14px;
    background: var(--bg);
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .no-calls {{
    font-size: 11px;
    color: var(--text3);
    font-style: italic;
  }}

  /* ── Call groups ── */
  .call-group {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    min-width: 160px;
    max-width: 300px;
    flex: 1 1 180px;
  }}
  .call-group-header {{
    padding: 5px 10px;
    background: var(--bg3);
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    font-weight: 600;
  }}
  .mod-link {{
    color: var(--accent);
    text-decoration: none;
    font-family: var(--font-mono);
  }}
  .mod-link:hover {{ text-decoration: underline; }}
  .mod-local {{ color: var(--text3); font-family: var(--font-mono); }}
  .call-list {{
    list-style: none;
    padding: 4px 0;
  }}
  .call-list li {{
    padding: 2px 10px;
    display: flex;
    align-items: baseline;
    gap: 4px;
    font-size: 12px;
    border-left: 2px solid transparent;
    transition: background 0.1s;
  }}
  .call-list li:hover {{ background: var(--bg3); border-left-color: var(--accent2); }}
  code.call-name {{
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text);
    background: none;
    flex: 1;
  }}
  .lineno {{
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text3);
    flex-shrink: 0;
  }}

  /* ── Docstring ── */
  .proc-doc {{
    padding: 5px 14px 6px;
    border-bottom: 1px solid var(--border);
    background: rgba(210,153,34,0.05);
  }}
  .doc-summary {{
    font-size: 12px;
    color: var(--text2);
    font-style: italic;
  }}
  .doc-details {{
    margin-top: 4px;
  }}
  .doc-details summary {{
    font-size: 11px;
    color: var(--text3);
    cursor: pointer;
    user-select: none;
  }}
  .doc-details summary:hover {{ color: var(--text2); }}
  .doc-rest {{
    margin-top: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text2);
    white-space: pre-wrap;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 8px;
    line-height: 1.5;
  }}

  /* ── Callers ── */
  .callers-section {{
    padding: 6px 14px 8px;
    border-top: 1px dashed var(--border);
    background: rgba(88,166,255,0.03);
  }}
  .callers-label {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--accent);
    opacity: 0.7;
    margin-bottom: 4px;
  }}
  .callers-list {{
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }}
  .callers-list li {{
    display: flex;
    align-items: baseline;
    gap: 5px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent2);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
  }}
  .caller-link {{
    text-decoration: none;
    color: var(--accent);
    font-family: var(--font-mono);
  }}
  .caller-link:hover {{ text-decoration: underline; }}
  .caller-file {{
    color: var(--text3);
    font-size: 10px;
    font-family: var(--font-mono);
  }}

  /* ── Hidden (search filter) ── */
  .hidden {{ display: none !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--text3); }}
</style>
</head>
<body>

<nav id="sidebar">
  <div class="sidebar-head">
    <div class="sidebar-project">📦 {_he(project_name)}</div>
    <div class="sidebar-meta">{len(files)} файлов · {total_procs} процедур · {total_calls} вызовов</div>
  </div>
  <div class="sidebar-search">
    <input type="text" id="sidebarSearch" placeholder="🔍 Файл или процедура..." autocomplete="off">
  </div>
  <div class="sidebar-files" id="sidebarFiles">
    {orphan_sidebar_link}{sidebar_html}
  </div>
  <div class="sidebar-gen">Сгенерировано {generated_at}</div>
</nav>

<main id="main">
  <h1 class="page-title">Карта вызовов — {_he(project_name)}</h1>
  <div class="page-meta">
    <span>📄 {len(files)} файлов</span>
    <span>⚙️ {total_procs} процедур</span>
    <span>🔗 {total_calls} вызовов</span>
    <span>🕐 {generated_at}</span>
  </div>

  <div class="main-search">
    <input type="text" id="mainSearch" placeholder="🔍 Поиск по процедурам и вызовам..." autocomplete="off">
  </div>

  {content_html}
</main>

<script>
  // ── Sidebar search ─────────────────────────────────────────────────────────
  const sidebarSearch = document.getElementById('sidebarSearch');
  sidebarSearch.addEventListener('input', () => {{
    const q = sidebarSearch.value.toLowerCase().trim();
    document.querySelectorAll('.sidebar-group').forEach(group => {{
      const fileName = group.querySelector('.sidebar-fname')?.textContent.toLowerCase() || '';
      let anyProcMatch = false;
      group.querySelectorAll('.sidebar-proc').forEach(proc => {{
        const match = !q || proc.textContent.toLowerCase().includes(q) || fileName.includes(q);
        proc.classList.toggle('hidden', !match);
        if (match) anyProcMatch = true;
      }});
      const groupMatch = !q || fileName.includes(q) || anyProcMatch;
      group.classList.toggle('hidden', !groupMatch);
      // Раскрыть группу если нашли процедуру
      if (q && anyProcMatch && !fileName.includes(q)) group.open = true;
    }});
  }});

  // ── Main search ────────────────────────────────────────────────────────────
  const mainSearch = document.getElementById('mainSearch');
  mainSearch.addEventListener('input', () => {{
    const q = mainSearch.value.toLowerCase().trim();
    document.querySelectorAll('.file-section').forEach(section => {{
      let sectionHasMatch = false;
      section.querySelectorAll('.proc-card').forEach(card => {{
        const text = card.textContent.toLowerCase();
        const match = !q || text.includes(q);
        card.classList.toggle('hidden', !match);
        if (match) sectionHasMatch = true;
      }});
      section.classList.toggle('hidden', q && !sectionHasMatch);
    }});
  }});

  // ── Active sidebar on scroll ──────────────────────────────────────────────
  const observer = new IntersectionObserver(entries => {{
    entries.forEach(entry => {{
      if (!entry.isIntersecting) return;
      const id = entry.target.id;

      // Подсветка файловой группы
      document.querySelectorAll('.sidebar-group').forEach(g => {{
        const active = g.dataset.fid === id;
        g.classList.toggle('active-group', active);
        if (active) {{
          g.open = true;
          g.scrollIntoView({{ block: 'nearest' }});
        }}
      }});

      // Подсветка процедуры
      document.querySelectorAll('.sidebar-proc').forEach(link => {{
        const href = link.getAttribute('href');
        const active = href && href.slice(1).startsWith(id + '-') || href === '#' + id;
        link.classList.toggle('active', active);
        if (active) link.scrollIntoView({{ block: 'nearest' }});
      }});
    }});
  }}, {{ rootMargin: '-5% 0px -75% 0px', threshold: 0 }});

  document.querySelectorAll('.file-section').forEach(s => observer.observe(s));
  document.querySelectorAll('.proc-card').forEach(s => observer.observe(s));

  // ── Collapse/expand proc body on header click ──────────────────────────────
  document.querySelectorAll('.proc-header').forEach(header => {{
    header.addEventListener('click', () => {{
      const body = header.nextElementSibling;
      if (body && body.classList.contains('proc-body')) {{
        const collapsed = body.style.display === 'none';
        body.style.display = collapsed ? '' : 'none';
        header.style.opacity = collapsed ? '1' : '0.6';
      }}
    }});
  }});
</script>
</body>
</html>
"""


# ─── CLI ─────────────────────────────────────────────────────────────────────


# ─── Рендер D3.js граф ────────────────────────────────────────────────────────

def _build_graph_data(files: list[FileInfo], module_index: dict[str, str]) -> dict:
    """
    Строит два набора данных для D3:
      - file_graph:  узлы=файлы, рёбра=файл→файл (по вызовам между ними)
      - func_graph:  узлы=функции, рёбра=функция→функция
    """
    # ── File graph ─────────────────────────────────────────────────────────────
    file_nodes: dict[str, dict] = {}
    file_edges_set: dict[tuple, int] = {}  # (src_rel, dst_rel) -> count

    for f in files:
        file_nodes[f.rel_path] = {
            "id":    f.rel_path,
            "label": f.rel_path,
            "procs": len(f.procedures),
            "calls": sum(len(p.calls) for p in f.procedures),
        }

    for f in files:
        for proc in f.procedures:
            for call in proc.calls:
                if not call.callee_module or call.callee_module == _VAR_:
                    continue
                target_rel = resolve_module_to_file(call.callee_module, module_index)
                if not target_rel or target_rel == f.rel_path:
                    continue
                key = (f.rel_path, target_rel)
                file_edges_set[key] = file_edges_set.get(key, 0) + 1

    file_graph = {
        "nodes": list(file_nodes.values()),
        "links": [
            {"source": src, "target": dst, "weight": w}
            for (src, dst), w in file_edges_set.items()
        ],
    }

    # ── Func graph ─────────────────────────────────────────────────────────────
    func_nodes: dict[str, dict] = {}
    func_edges_set: dict[tuple, int] = {}

    for f in files:
        for proc in f.procedures:
            nid = f"{f.rel_path}::{proc.qualname}"
            func_nodes[nid] = {
                "id":       nid,
                "label":    proc.qualname,
                "file":     f.rel_path,
                "lineno":   proc.lineno,
                "is_async": proc.is_async,
                "n_calls":   len(proc.calls),
                "n_callers": len(proc.callers),
                "docstring": proc.docstring.strip().splitlines()[0].strip() if proc.docstring else "",
            }

    for f in files:
        for proc in f.procedures:
            src_id = f"{f.rel_path}::{proc.qualname}"
            for call in proc.calls:
                if not call.callee_module or call.callee_module == _VAR_:
                    target_rel = f.rel_path
                else:
                    target_rel = resolve_module_to_file(call.callee_module, module_index)
                    if not target_rel:
                        continue

                # Ищем точное совпадение по имени функции в целевом файле
                dst_id = None
                for candidate_id in func_nodes:
                    cfile, cqual = candidate_id.split("::", 1)
                    if cfile == target_rel and (
                        cqual == call.callee_func or
                        cqual.endswith("." + call.callee_func) or
                        cqual == call.callee_func.split(".")[-1]
                    ):
                        dst_id = candidate_id
                        break

                if dst_id and dst_id != src_id:
                    key = (src_id, dst_id)
                    func_edges_set[key] = func_edges_set.get(key, 0) + 1

    func_graph = {
        "nodes": list(func_nodes.values()),
        "links": [
            {"source": src, "target": dst, "weight": w}
            for (src, dst), w in func_edges_set.items()
        ],
    }

    return {"file_graph": file_graph, "func_graph": func_graph}



def render_graph(files: list[FileInfo], module_index: dict[str, str], project_name: str, graph_params: dict) -> str:
    import json as _json
    from datetime import datetime

    graph_data  = _build_graph_data(files, module_index)
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    data_json   = _json.dumps(graph_data, ensure_ascii=False)
    graph_params_json = _json.dumps(graph_params, ensure_ascii=False)

    # Цвет по директории
    dirs = sorted(set(
        str(Path(n["id"]).parent) for n in graph_data["file_graph"]["nodes"]
    ))
    dir_colors = [
        "#58a6ff","#3fb950","#d29922","#bc8cff",
        "#f85149","#79c0ff","#56d364","#e3b341",
        "#d2a8ff","#ff7b72","#ffa657","#a5d6ff",
    ]
    dir_color_map = {d: dir_colors[i % len(dir_colors)] for i, d in enumerate(dirs)}
    color_map = {
        n["id"]: dir_color_map[str(Path(n["id"]).parent)]
        for n in graph_data["file_graph"]["nodes"]
    }
    color_map_json = _json.dumps(color_map)

    # Список файлов для фильтра
    file_list_json = _json.dumps(sorted(f.rel_path for f in files))

    total_funcs      = len(graph_data["func_graph"]["nodes"])
    total_func_edges = len(graph_data["func_graph"]["links"])

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Call Graph — {_he(project_name)}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
:root {{
  --bg:#0d1117; --bg2:#161b22; --bg3:#1c2128; --bg4:#21262d;
  --border:#30363d; --border2:#484f58;
  --text:#e6edf3; --text2:#8b949e; --text3:#6e7681;
  --accent:#58a6ff; --accent2:#388bfd;
  --green:#3fb950; --yellow:#d29922; --purple:#bc8cff; --red:#f85149;
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:"JetBrains Mono","Fira Code",Consolas,monospace;
  --panel:300px;
}}
html,body {{ width:100%; height:100%; overflow:hidden; background:var(--bg); color:var(--text); font-family:var(--font); }}

/* ── Top bar ── */
#topbar {{
  position:fixed; top:0; left:0; right:0; height:48px;
  background:var(--bg2); border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:10px; padding:0 14px; z-index:100;
  flex-shrink:0;
}}
.title {{ font-weight:700; font-size:14px; color:var(--accent); white-space:nowrap; }}
.sep   {{ color:var(--border2); }}
.mode-btn {{
  padding:4px 12px; border-radius:20px; border:1px solid var(--border);
  background:transparent; color:var(--text2); font-size:12px; cursor:pointer;
  transition:all .15s; font-family:var(--font); white-space:nowrap;
}}
.mode-btn:hover  {{ background:var(--bg3); color:var(--text); }}
.mode-btn.active {{ background:var(--accent); border-color:var(--accent); color:#0d1117; font-weight:600; }}
.stats {{ font-size:11px; color:var(--text3); white-space:nowrap; }}

#fileFilter {{
  background:var(--bg3); border:1px solid var(--border); border-radius:6px;
  color:var(--text); font-size:12px; padding:4px 8px; outline:none;
  font-family:var(--font); max-width:180px; cursor:pointer;
}}
#fileFilter:focus {{ border-color:var(--accent); }}
#fileFilter option {{ background:var(--bg2); }}

#searchBox {{
  margin-left:auto; background:var(--bg3); border:1px solid var(--border);
  border-radius:6px; color:var(--text); font-size:12px; padding:5px 10px;
  outline:none; width:180px; font-family:var(--font);
}}
#searchBox:focus  {{ border-color:var(--accent); }}
#searchBox::placeholder {{ color:var(--text3); }}

/* ── Tuning ── */
#tuning {{
  display:flex; align-items:center; gap:10px; margin-left:14px;
  padding:4px 8px; background:var(--bg3); border:1px solid var(--border);
  border-radius:8px; font-size:11px; color:var(--text3);
}}
.t-item {{ display:flex; align-items:center; gap:6px; white-space:nowrap; }}
.t-item label {{ color:var(--text2); }}
.t-item input[type="range"] {{
  width:90px; accent-color:var(--accent); cursor:pointer;
}}
.t-val {{ min-width:32px; text-align:right; color:var(--text); font-family:var(--mono); }}
.t-btn {{
  border:1px solid var(--border); background:transparent; color:var(--text2);
  padding:2px 8px; border-radius:6px; cursor:pointer; font-size:11px;
}}
.t-btn:hover {{ background:var(--bg4); color:var(--text); }}

/* ── Layout ── */
#layout {{ position:fixed; top:48px; left:0; right:0; bottom:0; display:flex; }}
#canvas {{ flex:1; position:relative; overflow:hidden; }}
#canvas svg {{ width:100%; height:100%; }}

/* ── Side panel ── */
#panel {{
  width:var(--panel); min-width:var(--panel);
  background:var(--bg2); border-left:1px solid var(--border);
  display:flex; flex-direction:column; overflow:hidden;
  transition:width .2s;
}}
#panel.collapsed {{ width:0; min-width:0; border:none; }}
#panel-head {{
  padding:12px 14px 8px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; justify-content:space-between; flex-shrink:0;
}}
#panel-title {{ font-size:12px; font-weight:600; color:var(--accent); font-family:var(--mono); word-break:break-all; }}
#panel-close {{
  background:none; border:none; color:var(--text3); cursor:pointer;
  font-size:16px; padding:0 2px; line-height:1; flex-shrink:0;
}}
#panel-close:hover {{ color:var(--text); }}
#panel-body {{ flex:1; overflow-y:auto; padding:10px 14px; font-size:12px; }}
.p-section {{ margin-bottom:12px; }}
.p-label {{
  font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.6px;
  color:var(--text3); margin-bottom:5px;
}}
.p-meta {{ color:var(--text2); margin:2px 0; }}
.p-meta span {{ color:var(--text); font-family:var(--mono); }}
.p-list {{ list-style:none; }}
.p-list li {{
  padding:3px 0; border-bottom:1px solid var(--border);
  display:flex; align-items:baseline; gap:6px;
}}
.p-list li:last-child {{ border:none; }}
.p-fn {{
  font-family:var(--mono); font-size:11px; color:var(--accent);
  cursor:pointer; text-decoration:none; flex:1;
}}
.p-fn:hover {{ text-decoration:underline; }}
.p-file {{ font-size:10px; color:var(--text3); font-family:var(--mono); }}
.p-empty {{ color:var(--text3); font-style:italic; font-size:11px; }}
.p-deselect {{ margin-left:auto; color:var(--text3); cursor:pointer; font-size:11px; padding:0 2px; flex-shrink:0; }}
.p-deselect:hover {{ color:var(--red); }}
.p-doc  {{ font-size:11px; color:var(--text2); font-style:italic; padding:6px 0 2px; line-height:1.5; border-bottom:1px solid var(--border); margin-bottom:6px; }}
.isolate-btn {{
  width:100%; margin-top:10px;
  background:var(--bg3); border:1px solid var(--border2);
  border-radius:6px; color:var(--text2); font-size:11px;
  padding:5px; cursor:pointer; font-family:var(--font); transition:all .15s;
}}
.isolate-btn:hover {{ background:var(--bg4); color:var(--text); border-color:var(--accent); }}
.isolate-btn.active {{ background:rgba(88,166,255,.15); border-color:var(--accent); color:var(--accent); }}

/* ── Nodes ── */
.node circle {{ stroke-width:1.5px; cursor:pointer; }}
.node text {{
  font-size:10px; fill:var(--text); pointer-events:none;
  font-family:var(--mono);
  paint-order:stroke;
  stroke:#0d1117; stroke-width:3px; stroke-linejoin:round;
}}
.node.async-node circle {{ stroke-dasharray:4,2; }}
.node.pinned circle {{ stroke-width:3px; }}
.node.dimmed circle {{ opacity:.12; }}
.node.dimmed text   {{ opacity:.08; }}
.node.selected circle {{ stroke-width:3px; filter:drop-shadow(0 0 8px currentColor); }}
.node.highlighted circle {{ filter:drop-shadow(0 0 5px currentColor); }}
.node.ghost circle {{ opacity:.32; stroke-dasharray:3,3; }}
.node.ghost text   {{ opacity:.38; font-size:9px; }}
.node.ghost        {{ cursor:default; }}

/* ── Links ── */
.link {{
  stroke:var(--border2); stroke-opacity:.5; fill:none;
  marker-end:url(#arrow);
}}
.link.dimmed        {{ stroke-opacity:.04; }}
.link.highlight-out {{ stroke:var(--green);  stroke-opacity:1; stroke-width:2px; marker-end:url(#arrow-green); }}
.link.highlight-in  {{ stroke:var(--accent); stroke-opacity:1; stroke-width:2px; marker-end:url(#arrow-blue); }}

/* ── Hull (file group) ── */
.hull {{ fill-opacity:.06; stroke-opacity:.3; stroke-width:1.5; stroke-dasharray:5,3; }}

/* ── Hull labels ── */
.hull-label {{
  font-size:9px; fill:var(--text3); pointer-events:none;
  font-family:var(--mono); text-anchor:middle;
}}

/* ── Tooltip ── */
#tooltip {{
  position:fixed; pointer-events:none; display:none;
  background:var(--bg2); border:1px solid var(--border);
  border-radius:8px; padding:9px 13px; font-size:12px;
  max-width:260px; z-index:200;
  box-shadow:0 8px 24px rgba(0,0,0,.6);
}}
.tt-title {{ font-family:var(--mono); font-size:12px; font-weight:600; color:var(--accent); margin-bottom:5px; word-break:break-all; }}
.tt-row   {{ color:var(--text2); margin:2px 0; font-size:11px; }}
.tt-row span {{ color:var(--text); }}
.tt-doc {{ font-size:11px; color:var(--text2); font-style:italic; margin-top:5px; padding-top:5px; border-top:1px solid var(--border); line-height:1.4; }}

/* ── Legend ── */
#legend {{
  position:absolute; bottom:16px; left:16px;
  background:var(--bg2); border:1px solid var(--border);
  border-radius:8px; padding:9px 13px; font-size:11px; z-index:10;
}}
.leg-title {{ color:var(--text3); font-size:10px; text-transform:uppercase; letter-spacing:.6px; margin-bottom:5px; }}
.leg-row   {{ display:flex; align-items:center; gap:7px; margin:2px 0; color:var(--text2); }}
.leg-dot   {{ width:10px; height:10px; border-radius:50%; border:1.5px solid; flex-shrink:0; }}
.leg-line  {{ width:22px; height:2px; border-radius:1px; }}

/* ── Controls ── */
#controls {{
  position:absolute; bottom:16px; right:16px;
  display:flex; flex-direction:column; gap:6px; z-index:10;
}}
.ctrl-btn {{
  width:32px; height:32px; border-radius:6px;
  background:var(--bg2); border:1px solid var(--border);
  color:var(--text); font-size:15px; cursor:pointer;
  display:flex; align-items:center; justify-content:center;
  transition:background .1s;
}}
.ctrl-btn:hover {{ background:var(--bg3); }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:var(--border2); border-radius:3px; }}
</style>
</head>
<body>

<div id="topbar">
  <span class="title">📦 {_he(project_name)}</span>
  <span class="sep">|</span>
  <button class="mode-btn" id="btnFile" onclick="setMode('file')">Файл → Файл</button>
  <button class="mode-btn active" id="btnFunc" onclick="setMode('func')">Функция → Функция</button>
  <span class="sep">|</span>
  <select id="fileFilter" onchange="applyFileFilter()" title="Фильтр по файлу">
    <option value="">📂 Все файлы</option>
  </select>
  <span class="sep">|</span>
  <span class="stats" id="statsLabel">{total_funcs} функций · {total_func_edges} вызовов</span>
  <div id="tuning" title="Настройка раскладки (только для текущего режима)">
    <div class="t-item"><label>dist</label><input id="tDist" type="range" min="60" max="320" step="5"><span id="vDist" class="t-val"></span></div>
    <div class="t-item"><label>link</label><input id="tLink" type="range" min="0.05" max="0.9" step="0.01"><span id="vLink" class="t-val"></span></div>
    <div class="t-item"><label>charge</label><input id="tCharge" type="range" min="-2000" max="-50" step="10"><span id="vCharge" class="t-val"></span></div>
    <div class="t-item"><label>pad</label><input id="tPad" type="range" min="0" max="40" step="1"><span id="vPad" class="t-val"></span></div>
    <div class="t-item"><label>iter</label><input id="tIters" type="range" min="1" max="6" step="1"><span id="vIters" class="t-val"></span></div>
    <button class="t-btn" onclick="resetTuning()">reset</button>
    <button class="t-btn" onclick="copyCli()">copy all</button>
    <button class="t-btn" onclick="copyCliMode()">copy mode</button>
  </div>
  <input id="searchBox" type="text" placeholder="🔍 Поиск функции..." autocomplete="off">
</div>

<div id="layout">
  <div id="canvas">
    <svg id="svg"><g id="root"></g></svg>
    <div id="legend">
      <div class="leg-title">Узлы</div>
      <div class="leg-row"><div class="leg-dot" style="border-color:var(--green);background:var(--green)22"></div>нет входящих (root)</div>
      <div class="leg-row"><div class="leg-dot" style="border-color:var(--accent);background:var(--accent)22"></div>обычная функция</div>
      <div class="leg-row"><div class="leg-dot" style="border-color:var(--purple);background:var(--purple)22;border-style:dashed"></div>async функция</div>
      <div style="margin-top:6px" class="leg-title">Рёбра</div>
      <div class="leg-row"><div class="leg-line" style="background:var(--green)"></div>исходящий вызов</div>
      <div class="leg-row"><div class="leg-line" style="background:var(--accent)"></div>входящий вызов</div>
    </div>
    <div id="controls">
      <button class="ctrl-btn" onclick="zoomIn()"    title="Приблизить">＋</button>
      <button class="ctrl-btn" onclick="zoomOut()"   title="Отдалить">－</button>
      <button class="ctrl-btn" onclick="resetZoom()" title="Сброс">⌂</button>
      <button class="ctrl-btn" id="btnHulls" onclick="toggleHulls()" title="Группы файлов">▦</button>
    </div>
  </div>

  <div id="panel" class="collapsed">
    <div id="panel-head">
      <div id="panel-title">—</div>
      <button id="panel-close" onclick="closePanel()">✕</button>
    </div>
    <div id="panel-body"></div>
  </div>
</div>

<div id="tooltip"></div>

<script>
// ── Data ────────────────────────────────────────────────────────────────────
const DATA      = {data_json};
const GRAPH_CFG = {graph_params_json};
const cfg = {{ ...GRAPH_CFG }};
const COLOR_MAP = {color_map_json};
const FILE_LIST = {file_list_json};

// ── State ───────────────────────────────────────────────────────────────────
let currentMode    = 'func';
let simulation     = null;
let zoomBehavior   = null;
let svgSel, rootG;
let linkSel, nodeSel, hullSel, hullLabelSel;
let showHulls      = true;
let isolatedNode   = null;
let selectedNodes  = new Set();   // ids выбранных узлов (ctrl+click)
let currentNodes   = [];
let currentLinks   = [];
let nodeEdges      = {{}};

// ── Tuning (UI) ─────────────────────────────────────────────────────────────
function _modeKey(base) {{
  return base + (currentMode === 'file' ? '_file' : '_func');
}}

function _setVal(id, v) {{
  const el = document.getElementById(id);
  const valEl = document.getElementById('v' + id.slice(1));
  if (!el || !valEl) return;
  el.value = v;
  valEl.textContent = (typeof v === 'number' && !Number.isInteger(v)) ? v.toFixed(2) : v;
}}

function updateTuningUI() {{
  _setVal('tDist',   cfg[_modeKey('link_distance')]);
  _setVal('tLink',   cfg[_modeKey('link_strength')]);
  _setVal('tCharge', cfg[_modeKey('charge')]);
  _setVal('tPad',    cfg.collide_pad);
  _setVal('tIters',  cfg.collide_iters);
}}

function readTuningUI() {{
  const dist   = parseFloat(document.getElementById('tDist').value);
  const link   = parseFloat(document.getElementById('tLink').value);
  const charge = parseFloat(document.getElementById('tCharge').value);
  const pad    = parseFloat(document.getElementById('tPad').value);
  const iters  = parseInt(document.getElementById('tIters').value, 10);

  cfg[_modeKey('link_distance')] = dist;
  cfg[_modeKey('link_strength')] = link;
  cfg[_modeKey('charge')]        = charge;
  cfg.collide_pad   = pad;
  cfg.collide_iters = iters;
}}

function applyTuning() {{
  if (!simulation) return;
  const linkDistance   = cfg[_modeKey('link_distance')];
  const linkStrength   = cfg[_modeKey('link_strength')];
  const chargeStrength = cfg[_modeKey('charge')];
  const collidePad     = cfg.collide_pad;
  const collideIters   = cfg.collide_iters;

  const linkForce = simulation.force('link');
  if (linkForce) linkForce.distance(linkDistance).strength(linkStrength);
  const chargeForce = simulation.force('charge');
  if (chargeForce) chargeForce.strength(chargeStrength);
  const collideForce = simulation.force('collide');
  if (collideForce) collideForce.radius(d => rScale((d.n_calls||0)+(d.n_callers||0)) + collidePad)
                              .iterations(collideIters);
  simulation.alpha(0.9).restart();
}}

function resetTuning() {{
  Object.assign(cfg, GRAPH_CFG);
  updateTuningUI();
  applyTuning();
}}

function wireTuning() {{
  ['tDist','tLink','tCharge','tPad','tIters'].forEach(id => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {{
      readTuningUI();
      updateTuningUI();
      applyTuning();
    }});
  }});
  updateTuningUI();
}}

function buildCliCommand() {{
  // Собираем полный набор параметров, чтобы можно было зафиксировать
  // и FILE, и FUNC значения сразу.
  const parts = [
    'python callmap_gen.py .',
    '--format graph',
    `--graph-link-dist-file ${cfg.link_distance_file}`,
    `--graph-link-dist-func ${cfg.link_distance_func}`,
    `--graph-link-strength-file ${cfg.link_strength_file.toFixed(2)}`,
    `--graph-link-strength-func ${cfg.link_strength_func.toFixed(2)}`,
    `--graph-charge-file ${cfg.charge_file}`,
    `--graph-charge-func ${cfg.charge_func}`,
    `--graph-collide-pad ${cfg.collide_pad}`,
    `--graph-collide-iters ${cfg.collide_iters}`,
  ];
  return parts.join(' ');
}}

function buildCliCommandMode() {{
  const parts = [
    'python callmap_gen.py .',
    '--format graph',
  ];
  if (currentMode === 'file') {{
    parts.push(`--graph-link-dist-file ${cfg.link_distance_file}`);
    parts.push(`--graph-link-strength-file ${cfg.link_strength_file.toFixed(2)}`);
    parts.push(`--graph-charge-file ${cfg.charge_file}`);
  }} else {{
    parts.push(`--graph-link-dist-func ${cfg.link_distance_func}`);
    parts.push(`--graph-link-strength-func ${cfg.link_strength_func.toFixed(2)}`);
    parts.push(`--graph-charge-func ${cfg.charge_func}`);
  }}
  parts.push(`--graph-collide-pad ${cfg.collide_pad}`);
  parts.push(`--graph-collide-iters ${cfg.collide_iters}`);
  return parts.join(' ');
}}

function copyCli() {{
  const cmd = buildCliCommand();
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(cmd).then(() => {{
      const btns = document.querySelectorAll('#tuning .t-btn');
      const btn = btns[btns.length - 2];
      if (btn) {{
        const prev = btn.textContent;
        btn.textContent = 'copied';
        setTimeout(() => btn.textContent = prev, 1200);
      }}
    }}).catch(() => {{
      prompt('Скопируйте команду:', cmd);
    }});
  }} else {{
    prompt('Скопируйте команду:', cmd);
  }}
}}

function copyCliMode() {{
  const cmd = buildCliCommandMode();
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(cmd).then(() => {{
      const btns = document.querySelectorAll('#tuning .t-btn');
      const btn = btns[btns.length - 1];
      if (btn) {{
        const prev = btn.textContent;
        btn.textContent = 'copied';
        setTimeout(() => btn.textContent = prev, 1200);
      }}
    }}).catch(() => {{
      prompt('Скопируйте команду:', cmd);
    }});
  }} else {{
    prompt('Скопируйте команду:', cmd);
  }}
}}

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {{
  svgSel = d3.select('#svg');
  rootG  = d3.select('#root');

  // Arrow markers
  const defs = svgSel.append('defs');
  function addArrow(id, color) {{
    defs.append('marker').attr('id', id)
      .attr('viewBox','0 -4 8 8').attr('refX',14).attr('refY',0)
      .attr('markerWidth',6).attr('markerHeight',6).attr('orient','auto')
      .append('path').attr('d','M0,-4L8,0L0,4').attr('fill', color);
  }}
  addArrow('arrow',       '#484f58');
  addArrow('arrow-green', '#3fb950');
  addArrow('arrow-blue',  '#58a6ff');

  zoomBehavior = d3.zoom().scaleExtent([0.03, 5])
    .on('zoom', e => rootG.attr('transform', e.transform));
  svgSel.call(zoomBehavior)
    .on('dblclick.zoom', null)
    .on('click', (e) => {{ if (!e.ctrlKey && !e.metaKey) deselect(); }});

  // File filter dropdown
  const sel = document.getElementById('fileFilter');
  FILE_LIST.forEach(f => {{
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = '📄 ' + f;
    sel.appendChild(opt);
  }});

  document.getElementById('searchBox').addEventListener('input', onSearch);
  wireTuning();

  drawGraph('func');
}});

// ── Draw ─────────────────────────────────────────────────────────────────────
function drawGraph(mode) {{
  rootG.selectAll('*').remove();
  if (simulation) simulation.stop();
  isolatedNode  = null;
  selectedNodes = new Set();
  closePanel();

  const raw = mode === 'file' ? DATA.file_graph : DATA.func_graph;
  const filterFile = document.getElementById('fileFilter').value;

  // Clone + filter
  const allNodes = raw.nodes.map(d => ({{...d}}));

  let nodes, ghostIds;
  if (filterFile && mode === 'func') {{
    const primaryIds = new Set(allNodes.filter(n => n.file === filterFile).map(n => n.id));
    ghostIds = new Set();
    raw.links.forEach(l => {{
      if (primaryIds.has(l.source) && !primaryIds.has(l.target)) ghostIds.add(l.target);
      if (primaryIds.has(l.target) && !primaryIds.has(l.source)) ghostIds.add(l.source);
    }});
    nodes = allNodes
      .filter(n => primaryIds.has(n.id) || ghostIds.has(n.id))
      .map(n => ({{ ...n, _ghost: ghostIds.has(n.id) }}));
  }} else {{
    ghostIds = new Set();
    nodes = allNodes;
  }}

  const nodeIds = new Set(nodes.map(n => n.id));
  let links = raw.links
    .filter(l => nodeIds.has(l.source) && nodeIds.has(l.target))
    .map(l => ({{...l}}));

  currentNodes = nodes;
  currentLinks = links;

  const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
  links = links.map(l => ({{ ...l, source: nodeById[l.source], target: nodeById[l.target] }}));

  const primaryCount = nodes.filter(n => !n._ghost).length;
  const ghostCount   = nodes.filter(n =>  n._ghost).length;
  document.getElementById('statsLabel').textContent = ghostCount > 0
    ? primaryCount + ' узлов · ' + ghostCount + ' внешних · ' + links.length + ' рёбер'
    : nodes.length + ' узлов · ' + links.length + ' рёбер';

  const W = document.getElementById('canvas').clientWidth;
  const H = document.getElementById('canvas').clientHeight;

  // Scales
  const maxDeg = Math.max(1, ...nodes.map(n => (n.n_calls||0) + (n.n_callers||0)));
  const rScale = d3.scaleSqrt().domain([0, maxDeg]).range(mode === 'file' ? [12,36] : [5,18]);

  function nodeColor(n) {{
    if (mode === 'file') return COLOR_MAP[n.id]  || '#58a6ff';
    if (n.is_async)      return '#bc8cff';
    if ((n.n_callers||0) === 0) return '#3fb950';  // root node
    return COLOR_MAP[n.file] || '#58a6ff';
  }}

  // ── Hull layer (file grouping) ─────────────────────────────────────────────
  hullSel      = rootG.append('g').attr('class','hulls');
  hullLabelSel = rootG.append('g').attr('class','hull-labels');

  // ── Link layer ─────────────────────────────────────────────────────────────
  const linkG = rootG.append('g');
  linkSel = linkG.selectAll('line').data(links).join('line')
    .attr('class','link')
    .attr('stroke-width', d => Math.min(4, 0.8 + (d.weight||1) * 0.5));

  // ── Node layer ─────────────────────────────────────────────────────────────
  nodeSel = rootG.append('g').selectAll('g.node').data(nodes).join('g')
    .attr('class', d => 'node' + (d.is_async ? ' async-node' : '') + (d._ghost ? ' ghost' : ''))
    .call(d3.drag()
      .on('start', (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
      .on('drag',  (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
      .on('end',   (e,d) => {{
        if(!e.active) simulation.alphaTarget(0);
        if(e.sourceEvent.type==='mouseup' && !e.sourceEvent.defaultPrevented) {{}}
        // keep pinned
      }}))
    .on('mouseover', (e,d) => onHover(e,d))
    .on('mouseout',  ()    => onOut())
    .on('click',     (e,d) => {{ e.stopPropagation(); if (!d._ghost) onNodeClick(d, e); }});

  nodeSel.append('circle')
    .attr('r', d => rScale((d.n_calls||0) + (d.n_callers||0)))
    .attr('fill',   d => nodeColor(d) + '22')
    .attr('stroke', d => nodeColor(d));

  nodeSel.append('text')
    .attr('dy', d => rScale((d.n_calls||0)+(d.n_callers||0)) + 12)
    .attr('text-anchor','middle')
    .text(d => truncate(d.label, mode==='file'?28:22));

  // ── Индекс рёбер по узлу (для синхронизации тултипа и панели) ───────────────
  nodeEdges = {{}};
  nodes.forEach(n => nodeEdges[n.id] = {{ out: [], in: [] }});
  links.forEach(l => {{
    const sid = typeof l.source === 'object' ? l.source.id : l.source;
    const tid = typeof l.target === 'object' ? l.target.id : l.target;
    if (nodeEdges[sid]) nodeEdges[sid].out.push(l);
    if (nodeEdges[tid]) nodeEdges[tid].in.push(l);
  }});

  // ── Simulation ─────────────────────────────────────────────────────────────
  // Разделяем ghost и primary для жёсткого клампинга в ticked()
  const _primaryNodes = (filterFile && mode === 'func') ? nodes.filter(n => !n._ghost) : [];
  const _ghostNodes   = (filterFile && mode === 'func') ? nodes.filter(n =>  n._ghost) : [];

  // Вычислить радиус hull в направлении угла theta:
  // hull — массив точек полигона; возвращает расстояние от centroid до границы.
  function hullRadiusAt(hull, cx, cy, theta) {{
    if (!hull || hull.length < 2) return 60;
    const cos = Math.cos(theta), sin = Math.sin(theta);
    let minT = Infinity;
    for (let i = 0; i < hull.length; i++) {{
      const [x1, y1] = hull[i];
      const [x2, y2] = hull[(i+1) % hull.length];
      const ex = x2-x1, ey = y2-y1;
      const denom = cos*ey - sin*ex;
      if (Math.abs(denom) < 1e-9) continue;
      const t = ((x1-cx)*ey - (y1-cy)*ex) / denom;
      const u = ((x1-cx)*sin - (y1-cy)*cos) / denom;  // recheck sign
      if (t > 0 && u >= 0 && u <= 1) minT = Math.min(minT, t);
    }}
    return isFinite(minT) ? minT : 60;
  }}

  // Параметры сил подравниваем так, чтобы узлы распределялись ровнее
  // и меньше кучковались в плотные "шары".
  const linkDistance   = mode === 'file' ? cfg.link_distance_file : cfg.link_distance_func;
  const linkStrength   = mode === 'file' ? cfg.link_strength_file : cfg.link_strength_func;
  const chargeStrength = mode === 'file' ? cfg.charge_file : cfg.charge_func;
  const collideRadius  = d => rScale((d.n_calls||0)+(d.n_callers||0)) + cfg.collide_pad;
  const collideIters   = cfg.collide_iters;

  simulation = d3.forceSimulation(nodes)
    .force('link',    d3.forceLink(links).id(d=>d.id).distance(linkDistance).strength(linkStrength))
    .force('charge',  d3.forceManyBody().strength(chargeStrength))
    .force('center',  d3.forceCenter(W/2, H/2))
    .force('collide', d3.forceCollide().radius(collideRadius).iterations(collideIters))
    .on('tick', ticked);

  function ticked() {{
    // ── Жёсткий клампинг ghost за пределы hull ───────────────────────────────
    // На каждом тике: если ghost оказался внутри hull + margin — выталкиваем.
    // margin=28 совпадает с hull pad в drawHulls, поэтому ghost прижимаются
    // вплотную к видимой оболочке, но не залезают за неё.
    if (_ghostNodes.length && _primaryNodes.length) {{
      const cx = d3.mean(_primaryNodes, n => n.x);
      const cy = d3.mean(_primaryNodes, n => n.y);
      const pts = _primaryNodes.map(n => [n.x, n.y]);
      const convex = pts.length >= 3 ? d3.polygonHull(pts) : null;
      const margin = 32;  // отступ от края hull до ghost

      _ghostNodes.forEach(n => {{
        const dx = n.x - cx, dy = n.y - cy;
        const dist = Math.sqrt(dx*dx + dy*dy) || 1;
        const theta = Math.atan2(dy, dx);

        // Радиус hull в направлении этого ghost + margin
        let hullR;
        if (convex) {{
          // Расширяем hull точно так же как в drawHulls (pad=28)
          const expandedHull = convex.map(([x,y]) => {{
            const ex=x-cx, ey=y-cy, ed=Math.sqrt(ex*ex+ey*ey)||1;
            return [cx+ex/ed*(ed+28), cy+ey/ed*(ed+28)];
          }});
          hullR = hullRadiusAt(expandedHull, cx, cy, theta) + margin;
        }} else {{
          // Мало точек — используем maxR
          hullR = Math.max(40, d3.max(_primaryNodes, p => Math.sqrt((p.x-cx)**2+(p.y-cy)**2)) || 40) + 28 + margin;
        }}

        if (dist < hullR) {{
          const scale = hullR / dist;
          n.x = cx + dx * scale;
          n.y = cy + dy * scale;
          const dot = n.vx*(dx/dist) + n.vy*(dy/dist);
          if (dot < 0) {{ n.vx -= dot*(dx/dist); n.vy -= dot*(dy/dist); }}
        }}
      }});
    }}

    linkSel
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => edgeEnd(d.source, d.target, rScale((d.target.n_calls||0)+(d.target.n_callers||0))).x)
      .attr('y2', d => edgeEnd(d.source, d.target, rScale((d.target.n_calls||0)+(d.target.n_callers||0))).y);
    nodeSel.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
    if (showHulls && mode === 'func') drawHulls(nodes);
  }}
}}

// ── Hull drawing ─────────────────────────────────────────────────────────────
function drawHulls(nodes) {{
  const byFile = d3.group(nodes.filter(n => !n._ghost), d => d.file || '');
  const hullData = [];
  byFile.forEach((fnodes, file) => {{
    if (!file || fnodes.length < 1) return;
    const pts = fnodes.map(n => [n.x, n.y]);
    const pad = 28;
    let hull;
    if (pts.length === 1) {{
      const [cx,cy] = pts[0];
      hull = [[cx-pad,cy-pad],[cx+pad,cy-pad],[cx+pad,cy+pad],[cx-pad,cy+pad]];
    }} else if (pts.length === 2) {{
      const [[x1,y1],[x2,y2]] = pts;
      const mx=(x1+x2)/2, my=(y1+y2)/2;
      hull = d3.polygonHull([[x1-pad,y1-pad],[x1+pad,y1-pad],[x2+pad,y2+pad],[x2-pad,y2+pad]]) || pts;
    }} else {{
      hull = d3.polygonHull(pts.map(([x,y]) => [x,y]));
      if (!hull) return;
      // expand hull
      const cx = d3.mean(hull, p=>p[0]);
      const cy = d3.mean(hull, p=>p[1]);
      hull = hull.map(([x,y]) => {{
        const dx=x-cx, dy=y-cy, dist=Math.sqrt(dx*dx+dy*dy)||1;
        return [cx+dx/dist*(dist+pad), cy+dy/dist*(dist+pad)];
      }});
    }}
    const color = (nodes.find(n=>n.file===file) ? COLOR_MAP[file] : null) || '#8b949e';
    // centroid for label
    const lx = d3.mean(hull, p=>p[0]);
    const ly = Math.min(...hull.map(p=>p[1])) - 8;
    hullData.push({{ file, hull, color, lx, ly }});
  }});

  hullSel.selectAll('path').data(hullData).join('path')
    .attr('class','hull')
    .attr('d', d => 'M' + d.hull.map(p=>p.join(',')).join('L') + 'Z')
    .attr('fill',   d => d.color)
    .attr('stroke', d => d.color);

  hullLabelSel.selectAll('text').data(hullData).join('text')
    .attr('class','hull-label')
    .attr('x', d => d.lx).attr('y', d => d.ly)
    .text(d => truncate(d.file, 30));
}}

// ── Hover ─────────────────────────────────────────────────────────────────────
function onHover(e, d) {{
  if (isolatedNode || selectedNodes.size > 0) return;
  highlightNode(d.id);

  const edges = nodeEdges[d.id] || {{ out: [], in: [] }};
  const tt = document.getElementById('tooltip');
  let h = `<div class="tt-title">${{d.label}}</div>`;
  if (d.file)   h += `<div class="tt-row">📄 <span>${{d.file}}</span></div>`;
  if (d.lineno) h += `<div class="tt-row">📌 строка <span>${{d.lineno}}</span></div>`;
  h += `<div class="tt-row">→ вызывает: <span>${{edges.out.length}}</span></div>`;
  h += `<div class="tt-row">← вызывается: <span>${{edges.in.length}}</span></div>`;
  if (d.is_async)  h += `<div class="tt-row">⚡ <span>async</span></div>`;
  if (d.docstring)  h += `<div class="tt-doc">${{d.docstring}}</div>`;
  if (d._ghost) {{
    h += `<div class="tt-row" style="margin-top:5px;color:var(--text3);font-size:10px">🔗 внешний файл</div>`;
  }} else {{
    h += `<div class="tt-row" style="margin-top:5px;color:var(--text3);font-size:10px">Клик — подробности</div>`;
  }}
  tt.innerHTML = h; tt.style.display = 'block';
  document.addEventListener('mousemove', moveTip);
  moveTip(e);
}}

function onOut() {{
  if (isolatedNode || selectedNodes.size > 0) return;
  unhighlight();
  document.getElementById('tooltip').style.display = 'none';
  document.removeEventListener('mousemove', moveTip);
}}

function moveTip(e) {{
  const tt = document.getElementById('tooltip');
  const x = e.clientX+16, y = e.clientY-10;
  tt.style.left = (x + tt.offsetWidth  > window.innerWidth  ? x - tt.offsetWidth  - 20 : x) + 'px';
  tt.style.top  = (y + tt.offsetHeight > window.innerHeight ? y - tt.offsetHeight     : y) + 'px';
}}

// ── Node click → side panel ────────────────────────────────────────────────
function onNodeClick(d, e) {{
  document.getElementById('tooltip').style.display = 'none';
  document.removeEventListener('mousemove', moveTip);

  const multi = e && (e.ctrlKey || e.metaKey);

  if (multi) {{
    // Ctrl+click: добавить/убрать из выбора
    if (selectedNodes.has(d.id)) {{
      selectedNodes.delete(d.id);
    }} else {{
      selectedNodes.add(d.id);
    }}
  }} else {{
    // Обычный клик: выбрать только этот узел
    if (selectedNodes.size === 1 && selectedNodes.has(d.id)) {{
      deselect(); return;
    }}
    selectedNodes = new Set([d.id]);
  }}

  if (selectedNodes.size === 0) {{ deselect(); return; }}

  nodeSel.classed('selected', n => selectedNodes.has(n.id));
  highlightMulti(selectedNodes);
  openPanel(d);
}}

function deselect() {{
  selectedNodes = new Set();
  nodeSel.classed('selected', false);
  if (!isolatedNode) unhighlight();
  closePanel();
}}

function highlightNode(id) {{
  highlightMulti(new Set([id]));
}}

function highlightMulti(ids) {{
  // Собираем всех соседей всех выбранных узлов
  const connectedIds = new Set(ids);
  linkSel.each(l => {{
    const s = l.source.id||l.source, t = l.target.id||l.target;
    if (ids.has(s)) connectedIds.add(t);
    if (ids.has(t)) connectedIds.add(s);
  }});
  nodeSel.classed('dimmed',      n => !connectedIds.has(n.id));
  nodeSel.classed('highlighted', n => connectedIds.has(n.id) && !ids.has(n.id));
  linkSel.classed('dimmed',       l => {{
    const s=l.source.id||l.source, t=l.target.id||l.target;
    return !ids.has(s) && !ids.has(t);
  }});
  linkSel.classed('highlight-out',l => {{ const s=l.source.id||l.source; return ids.has(s); }});
  linkSel.classed('highlight-in', l => {{ const t=l.target.id||l.target; return ids.has(t) && !ids.has(l.source.id||l.source); }});
}}

function unhighlight() {{
  nodeSel.classed('dimmed highlighted', false);
  linkSel.classed('dimmed highlight-out highlight-in', false);
}}

// ── Side panel ──────────────────────────────────────────────────────────────
function openPanel(d) {{
  const panel = document.getElementById('panel');
  panel.classList.remove('collapsed');

  if (selectedNodes.size > 1) {{
    // ── Мульти-режим: показываем все выбранные узлы ───────────────────────────
    document.getElementById('panel-title').textContent = `${{selectedNodes.size}} функции выбраны`;

    // Собираем объединённые callees/callers по всем выбранным
    const allCallees = new Map();   // id → node (дедупликация)
    const allCallers = new Map();
    selectedNodes.forEach(sid => {{
      const edges = nodeEdges[sid] || {{ out: [], in: [] }};
      edges.out.forEach(l => {{
        const t = typeof l.target==='object' ? l.target : currentNodes.find(n=>n.id===l.target);
        if (t && !selectedNodes.has(t.id)) allCallees.set(t.id, t);
      }});
      edges.in.forEach(l => {{
        const s = typeof l.source==='object' ? l.source : currentNodes.find(n=>n.id===l.source);
        if (s && !selectedNodes.has(s.id)) allCallers.set(s.id, s);
      }});
    }});

    let html = `<div class="p-section">`;
    html += `<div class="p-label">Выбранные функции (${{selectedNodes.size}})</div>`;
    html += `<ul class="p-list">`;
    selectedNodes.forEach(sid => {{
      const n = currentNodes.find(x => x.id === sid);
      if (n) html += `<li>
        <a class="p-fn" onclick="focusNode('${{n.id}}')">${{n.label}}()</a>
        <span class="p-file">${{n.file}}</span>
        <span class="p-deselect" onclick="deselectOne('${{n.id}}')" title="Убрать из выбора">✕</span>
      </li>`;
    }});
    html += `</ul></div>`;

    const calleesArr = [...allCallees.values()];
    const callersArr = [...allCallers.values()];

    html += `<div class="p-section"><div class="p-label">→ Вызывают вместе (${{calleesArr.length}})</div>`;
    if (calleesArr.length) {{
      html += `<ul class="p-list">` + calleesArr.map(n =>
        `<li><a class="p-fn" onclick="focusNode('${{n.id}}')">${{n.label}}()</a><span class="p-file">${{n.file}}</span></li>`
      ).join('') + `</ul>`;
    }} else html += `<div class="p-empty">нет общих исходящих</div>`;
    html += `</div>`;

    html += `<div class="p-section"><div class="p-label">← Вызываются из (${{callersArr.length}})</div>`;
    if (callersArr.length) {{
      html += `<ul class="p-list">` + callersArr.map(n =>
        `<li><a class="p-fn" onclick="focusNode('${{n.id}}')">${{n.label}}()</a><span class="p-file">${{n.file}}</span></li>`
      ).join('') + `</ul>`;
    }} else html += `<div class="p-empty">нет общих входящих</div>`;
    html += `</div>`;

    html += `<button class="isolate-btn" onclick="deselectAll()">✕ Сбросить выбор</button>`;
    document.getElementById('panel-body').innerHTML = html;

  }} else {{
    // ── Одиночный выбор ───────────────────────────────────────────────────────
    document.getElementById('panel-title').textContent = d.label + '()';

    const edges   = nodeEdges[d.id] || {{ out: [], in: [] }};
    const callees = edges.out.map(l => typeof l.target==='object' ? l.target : currentNodes.find(n=>n.id===l.target)).filter(Boolean);
    const callers = edges.in.map(l  => typeof l.source==='object' ? l.source : currentNodes.find(n=>n.id===l.source)).filter(Boolean);

    let html = `<div class="p-section">`;
    html += `<div class="p-label">📄 Файл</div>`;
    html += `<div class="p-meta"><span>${{d.file||'—'}}</span></div>`;
    if (d.lineno)    html += `<div class="p-meta">строка <span>${{d.lineno}}</span></div>`;
    if (d.is_async)  html += `<div class="p-meta"><span>async</span></div>`;
    if (d.docstring) html += `<div class="p-doc">${{d.docstring}}</div>`;
    html += `</div>`;

    html += `<div class="p-section"><div class="p-label">→ Вызывает (${{callees.length}})</div>`;
    if (callees.length) {{
      html += `<ul class="p-list">` + callees.map(n =>
        `<li><a class="p-fn" onclick="focusNode('${{n.id}}')">${{n.label}}()</a><span class="p-file">${{n.file}}</span></li>`
      ).join('') + `</ul>`;
    }} else html += `<div class="p-empty">нет исходящих вызовов</div>`;
    html += `</div>`;

    html += `<div class="p-section"><div class="p-label">← Вызывается из (${{callers.length}})</div>`;
    if (callers.length) {{
      html += `<ul class="p-list">` + callers.map(n =>
        `<li><a class="p-fn" onclick="focusNode('${{n.id}}')">${{n.label}}()</a><span class="p-file">${{n.file}}</span></li>`
      ).join('') + `</ul>`;
    }} else html += `<div class="p-empty">нет входящих вызовов</div>`;
    html += `</div>`;

    const isIsolated = isolatedNode && isolatedNode.id === d.id;
    html += `<button class="isolate-btn${{isIsolated?' active':''}}" id="isolateBtn" onclick="toggleIsolate('${{d.id}}')">`
          + (isIsolated ? '🔓 Показать всё' : '🔍 Изолировать функцию')
          + `</button>`;

    document.getElementById('panel-body').innerHTML = html;
  }}
}}

function closePanel() {{
  document.getElementById('panel').classList.add('collapsed');
}}

// ── Focus / Isolate ─────────────────────────────────────────────────────────
function deselectOne(id) {{
  selectedNodes.delete(id);
  if (selectedNodes.size === 0) {{ deselect(); return; }}
  nodeSel.classed('selected', n => selectedNodes.has(n.id));
  highlightMulti(selectedNodes);
  // Переоткрыть панель с последним выбранным
  const lastId = [...selectedNodes].at(-1);
  const last = currentNodes.find(n => n.id === lastId);
  if (last) openPanel(last);
}}

function deselectAll() {{
  deselect();
}}

function focusNode(id) {{
  const d = currentNodes.find(n => n.id === id);
  if (!d) return;
  selectedNodes = new Set([id]);
  nodeSel.classed('selected', n => n.id === id);
  highlightNode(id);
  openPanel(d);
  // Pan to node
  const W = document.getElementById('canvas').clientWidth;
  const H = document.getElementById('canvas').clientHeight;
  svgSel.transition().duration(500)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(W/2 - d.x, H/2 - d.y));
}}

function toggleIsolate(id) {{
  if (isolatedNode && isolatedNode.id === id) {{
    isolatedNode = null;
    unhighlight();
    nodeSel.classed('dimmed', false);
    linkSel.classed('dimmed', false);
    document.getElementById('isolateBtn').textContent = '🔍 Изолировать функцию';
    document.getElementById('isolateBtn').classList.remove('active');
  }} else {{
    const d = currentNodes.find(n => n.id === id);
    isolatedNode = d;
    highlightNode(id);
    document.getElementById('isolateBtn').textContent = '🔓 Показать всё';
    document.getElementById('isolateBtn').classList.add('active');
  }}
}}

// ── Search ───────────────────────────────────────────────────────────────────
function onSearch() {{
  const q = document.getElementById('searchBox').value.toLowerCase().trim();
  if (!q) {{ unhighlight(); return; }}
  nodeSel.classed('dimmed',      d => !d.label.toLowerCase().includes(q));
  nodeSel.classed('highlighted', d =>  d.label.toLowerCase().includes(q));
  linkSel.classed('dimmed', true);
}}

// ── File filter ──────────────────────────────────────────────────────────────
function applyFileFilter() {{
  drawGraph(currentMode);
  setTimeout(() => resetZoom(), 600);
}}

// ── Hulls toggle ─────────────────────────────────────────────────────────────
function toggleHulls() {{
  showHulls = !showHulls;
  document.getElementById('btnHulls').style.opacity = showHulls ? '1' : '0.4';
  if (!showHulls) {{
    hullSel.selectAll('*').remove();
    hullLabelSel.selectAll('*').remove();
  }}
}}

// ── Mode switch ──────────────────────────────────────────────────────────────
function setMode(mode) {{
  currentMode = mode;
  document.getElementById('btnFile').classList.toggle('active', mode==='file');
  document.getElementById('btnFunc').classList.toggle('active', mode==='func');
  document.getElementById('fileFilter').value = '';
  document.getElementById('searchBox').value  = '';
  document.getElementById('fileFilter').style.display = mode==='func' ? '' : 'none';
  updateTuningUI();
  drawGraph(mode);
  setTimeout(resetZoom, 700);
}}

// ── Zoom ─────────────────────────────────────────────────────────────────────
function zoomIn()    {{ svgSel.transition().call(zoomBehavior.scaleBy, 1.4); }}
function zoomOut()   {{ svgSel.transition().call(zoomBehavior.scaleBy, 0.7); }}
function resetZoom() {{
  const W = document.getElementById('canvas').clientWidth;
  const H = document.getElementById('canvas').clientHeight;
  svgSel.transition().duration(500)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(W/2,H/2).scale(0.9).translate(-W/2,-H/2));
}}

// ── Helpers ──────────────────────────────────────────────────────────────────
function truncate(s, n) {{ return s && s.length > n ? s.slice(0,n-1)+'…' : s; }}
function edgeEnd(src, tgt, r) {{
  const dx=tgt.x-src.x, dy=tgt.y-src.y, dist=Math.sqrt(dx*dx+dy*dy)||1;
  return {{ x: tgt.x-(dx/dist)*r, y: tgt.y-(dy/dist)*r }};
}}
</script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(
        description="Генератор карты вызовов функций для Python-проектов.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("project", help="Путь к корню проекта")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Файл вывода. Формат определяется по расширению: .md или .html (default: call_map.md)",
    )
    parser.add_argument(
        "--format",
        choices=["md", "html", "graph", "both", "all"],
        default=None,
        help="Явно задать формат: md, html, graph (D3 граф), both (md+html), all (все три)",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Папки для исключения через запятую (добавляются к стандартным)",
    )
    parser.add_argument(
        "--exclude-libs",
        default="",
        help="Библиотеки/модули для исключения из вызовов через запятую "
             "(например: nicegui,datetime,requests). "
             "Поддерживаются префиксы: nicegui исключит и nicegui.ui",
    )
    parser.add_argument(
        "--no-locals",
        action="store_true",
        help="Не включать вложенные функции (локальные closure)",
    )
    parser.add_argument(
        "--min-calls",
        type=int,
        default=0,
        help="Показывать только процедуры с >= N вызовами (default: 0 — все)",
    )
    # Параметры раскладки графа (D3)
    parser.add_argument(
        "--graph-link-dist-file",
        type=float,
        default=180,
        help="Длина рёбер в режиме FILE (default: 180)",
    )
    parser.add_argument(
        "--graph-link-dist-func",
        type=float,
        default=120,
        help="Длина рёбер в режиме FUNC (default: 120)",
    )
    parser.add_argument(
        "--graph-link-strength-file",
        type=float,
        default=0.40,
        help="Сила стягивания рёбер в режиме FILE (default: 0.40)",
    )
    parser.add_argument(
        "--graph-link-strength-func",
        type=float,
        default=0.32,
        help="Сила стягивания рёбер в режиме FUNC (default: 0.32)",
    )
    parser.add_argument(
        "--graph-charge-file",
        type=float,
        default=-900,
        help="Отталкивание узлов в режиме FILE (default: -900)",
    )
    parser.add_argument(
        "--graph-charge-func",
        type=float,
        default=-320,
        help="Отталкивание узлов в режиме FUNC (default: -320)",
    )
    parser.add_argument(
        "--graph-collide-pad",
        type=float,
        default=10,
        help="Доп. радиус коллизии (default: 10)",
    )
    parser.add_argument(
        "--graph-collide-iters",
        type=int,
        default=2,
        help="Итерации коллизии (default: 2)",
    )
    args = parser.parse_args()

    root = Path(args.project).resolve()
    if not root.is_dir():
        print(f"Ошибка: '{root}' не является директорией.", file=sys.stderr)
        sys.exit(1)

    # Определяем форматы и пути вывода
    fmt = args.format
    output_arg = args.output

    if fmt in ("both", "all"):
        base = Path(output_arg).with_suffix("") if output_arg else Path("call_map")
        outputs = [
            (base.with_suffix(".md"),    "md"),
            (base.with_suffix(".html"),  "html"),
        ]
        if fmt == "all":
            outputs.append((base.with_suffix(".graph.html"), "graph"))
    elif output_arg:
        ext = Path(output_arg).suffix.lower()
        if ext == ".html" and fmt == "graph":
            detected_fmt = "graph"
        elif ext == ".html":
            detected_fmt = "html"
        else:
            detected_fmt = "md"
        fmt = fmt or detected_fmt
        outputs = [(Path(output_arg), fmt)]
    else:
        fmt = fmt or "md"
        suffix = ".graph.html" if fmt == "graph" else f".{fmt}"
        outputs = [(Path(f"call_map{suffix}"), fmt)]

    extra_exclude = set(e.strip() for e in args.exclude.split(",") if e.strip())

    # Заполняем глобальный фильтр библиотек
    excluded_libs = set(e.strip().lower() for e in args.exclude_libs.split(",") if e.strip())
    _EXCLUDED_LIBS.clear()
    _EXCLUDED_LIBS.update(excluded_libs)

    print(f"📂 Проект:    {root}")
    for out_path, out_fmt in outputs:
        print(f"📄 Вывод:     {out_path}  [{out_fmt}]")
    if extra_exclude:
        print(f"🚫 Исключено (папки): {extra_exclude}")
    if excluded_libs:
        print(f"🚫 Исключено (библиотеки): {excluded_libs}")

    print("\n🔍 Сканирование файлов...")
    files = scan_project(root, extra_exclude)
    print(f"   Найдено файлов: {len(files)}")

    total_procs = sum(len(f.procedures) for f in files)
    total_calls = sum(len(p.calls) for f in files for p in f.procedures)
    print(f"   Процедур:       {total_procs}")
    print(f"   Вызовов:        {total_calls}")

    if args.min_calls > 0:
        for f in files:
            f.procedures = [p for p in f.procedures if len(p.calls) >= args.min_calls]
        print(f"   После фильтра (>={args.min_calls} вызовов): {sum(len(f.procedures) for f in files)} процедур")

    files = [f for f in files if f.procedures]
    module_index = _build_module_index(files)
    project_name = root.name

    print("\n🔗 Построение индекса вызывающих функций...")
    build_callers_index(files, module_index)
    total_callers = sum(len(p.callers) for f in files for p in f.procedures)
    print(f"   Связей caller→callee: {total_callers}")

    graph_params = {
        "link_distance_file": args.graph_link_dist_file,
        "link_distance_func": args.graph_link_dist_func,
        "link_strength_file": args.graph_link_strength_file,
        "link_strength_func": args.graph_link_strength_func,
        "charge_file": args.graph_charge_file,
        "charge_func": args.graph_charge_func,
        "collide_pad": args.graph_collide_pad,
        "collide_iters": args.graph_collide_iters,
    }

    for out_path, out_fmt in outputs:
        print(f"\n📝 Генерация {out_fmt.upper()}...")
        if out_fmt == "html":
            content = render_html(files, module_index, project_name)
        elif out_fmt == "graph":
            content = render_graph(files, module_index, project_name, graph_params)
        else:
            content = render_markdown(files, module_index, project_name)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        size_kb = out_path.stat().st_size / 1024
        print(f"   Готово: {out_path} ({size_kb:.1f} KB)")

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
