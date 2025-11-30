import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional

class SummaryDatabase:
    def __init__(self, db_path: str = "summary_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa la base de datos y crea las tablas necesarias."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    word_count INTEGER,
                    char_count INTEGER,
                    processing_time REAL,
                    method TEXT DEFAULT 'unknown',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Crear índices para búsquedas rápidas
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON summaries(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_method ON summaries(method)")
            
            # Intentar añadir columna chunks_data si no existe (migración simple)
            try:
                conn.execute("ALTER TABLE summaries ADD COLUMN chunks_data TEXT")
            except sqlite3.OperationalError:
                pass
            
            # Intentar añadir columna title si no existe
            try:
                conn.execute("ALTER TABLE summaries ADD COLUMN title TEXT")
            except sqlite3.OperationalError:
                pass
            
            # Intentar añadir columna tags si no existe
            try:
                conn.execute("ALTER TABLE summaries ADD COLUMN tags TEXT")
            except sqlite3.OperationalError:
                pass
                
            conn.commit()
    
    def save_summary(self, 
                    original_text: str, 
                    summary: str, 
                    word_count: int, 
                    char_count: int, 
                    processing_time: float, 
                    method: str = 'unknown',
                    chunks_data: str = None,
                    title: str = None,
                    tags: str = None) -> int:
        """Guarda un resumen en la base de datos."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Si no hay título, usar timestamp como fallback
        if not title:
            title = f"Resumen {timestamp}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO summaries 
                (timestamp, original_text, summary, word_count, char_count, processing_time, method, chunks_data, title, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, original_text, summary, word_count, char_count, processing_time, method, chunks_data, title, tags))
            return cursor.lastrowid
    
    def get_recent_summaries(self, limit: int = 10) -> List[Dict]:
        """Obtiene los resúmenes más recientes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM summaries 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def search_summaries(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Busca resúmenes que contengan los términos especificados.
        Busca en título, texto original, resumen y tags.
        """
        terms = query.strip().split()
        if not terms:
            return []
            
        conditions = []
        params = []
        
        for term in terms:
            # Cada término debe estar presente en al menos uno de los campos
            conditions.append("(title LIKE ? OR original_text LIKE ? OR summary LIKE ? OR tags LIKE ?)")
            like_term = f"%{term}%"
            params.extend([like_term, like_term, like_term, like_term])
            
        where_clause = " AND ".join(conditions)
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM summaries 
                WHERE {where_clause}
                ORDER BY created_at DESC 
                LIMIT ?
            """, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_summary_by_id(self, summary_id: int) -> Optional[Dict]:
        """Obtiene un resumen específico por ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM summaries WHERE id = ?", (summary_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def delete_summary(self, summary_id: int) -> bool:
        """Elimina un resumen por ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
            return cursor.rowcount > 0
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas generales del historial."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_summaries,
                    SUM(word_count) as total_words,
                    AVG(processing_time) as avg_processing_time,
                    MIN(created_at) as first_summary,
                    MAX(created_at) as latest_summary
                FROM summaries
            """)
            row = cursor.fetchone()
            return {
                'total_summaries': row[0] or 0,
                'total_words': row[1] or 0,
                'avg_processing_time': round(row[2] or 0, 2),
                'first_summary': row[3],
                'latest_summary': row[4]
            }
    
    def cleanup_old_summaries(self, keep_last: int = 100):
        """Mantiene solo los últimos N resúmenes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM summaries 
                WHERE id NOT IN (
                    SELECT id FROM summaries 
                    ORDER BY created_at DESC 
                    LIMIT ?
                )
            """, (keep_last,))
            conn.commit()
    
    def get_all_tags(self) -> List[str]:
        """Obtiene todas las etiquetas únicas usadas en los resúmenes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT tags FROM summaries WHERE tags IS NOT NULL AND tags != ''")
            all_tags = set()
            for row in cursor:
                tags_str = row[0]
                if tags_str:
                    # Separar por comas y limpiar espacios
                    tags = [t.strip() for t in tags_str.split(',')]
                    all_tags.update(tags)
            return sorted(list(all_tags))

    def filter_summaries(self, query: str = None, tags: List[str] = None, limit: int = 50) -> List[Dict]:
        """
        Filtra resúmenes por texto y/o etiquetas.
        """
        conditions = []
        params = []
        
        # Filtro de texto
        if query:
            terms = query.strip().split()
            for term in terms:
                conditions.append("(title LIKE ? OR original_text LIKE ? OR summary LIKE ? OR tags LIKE ?)")
                like_term = f"%{term}%"
                params.extend([like_term, like_term, like_term, like_term])
        
        # Filtro de etiquetas (AND logic: debe tener TODAS las etiquetas seleccionadas)
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM summaries 
                WHERE {where_clause}
                ORDER BY created_at DESC 
                LIMIT ?
            """, params)
            return [dict(row) for row in cursor.fetchall()]

    def export_to_csv(self, filepath: str):
        """Exporta todos los resúmenes a un archivo CSV."""
        import csv
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM summaries ORDER BY created_at DESC")
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                if cursor.description:
                    fieldnames = [description[0] for description in cursor.description]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in cursor:
                        writer.writerow(dict(row))