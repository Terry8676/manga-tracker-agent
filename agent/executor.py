"""Action executor.

Takes validated actions and runs them against the three domains: the user's
database rows, the external manga APIs, and file export/import. Every action
is wrapped so a failure in one does not abort the rest of the plan, and every
outcome is written to the audit log.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import Series, VALID_STATUSES, utcnow
from .logger import record_action
from .manga_api import MangaAPI, MangaAPIError
from .schemas import (
    AddSeries,
    CheckNewChapters,
    ExportList,
    GetRecommendations,
    ImportList,
    QueryLibrary,
    RateSeries,
    RemoveSeries,
    SearchManga,
    SetStatus,
    UpdateProgress,
)


@dataclass
class ActionResult:
    action: str
    status: str  # ok | error | rejected | cancelled
    summary: str
    data: Any = None
    #: Attached when the action produced a downloadable file.
    download: dict | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class ExecutionReport:
    results: list[ActionResult] = field(default_factory=list)

    @property
    def any_failed(self) -> bool:
        return any(r.status == "error" for r in self.results)

    @property
    def downloads(self) -> list[dict]:
        return [r.download for r in self.results if r.download]


class Executor:
    """Dispatches actions for exactly one user."""

    def __init__(
        self,
        session: Session,
        user_id: int,
        *,
        settings: Settings | None = None,
        api: MangaAPI | None = None,
        upload_provider: Callable[[], str | None] | None = None,
    ):
        self.session = session
        self.user_id = user_id
        self.settings = settings or get_settings()
        self.api = api or MangaAPI(self.settings)
        #: Returns the text of a CSV the user uploaded in the UI, if any.
        self.upload_provider = upload_provider or (lambda: None)

    # -- helpers ------------------------------------------------------------ #

    def _all_series(self) -> list[Series]:
        return (
            self.session.query(Series)
            .filter(Series.user_id == self.user_id)
            .order_by(Series.updated_at.desc())
            .all()
        )

    def _find_exact(self, title: str) -> Series | None:
        """Case-insensitive exact title match.

        Used for duplicate detection, where loose matching would wrongly
        reject a sequel or spin-off whose title contains an existing one
        (e.g. adding 'Attack on Titan: Before the Fall' when 'Attack on
        Titan' is already on the list).
        """
        wanted = title.strip().lower()
        for row in self._all_series():
            if row.title.strip().lower() == wanted:
                return row
        return None

    def _find(self, title: str) -> Series | None:
        """Locate an existing series to act on, tolerating loose titles.

        Exact match wins. Otherwise a substring match in either direction is
        accepted, but only when exactly one row matches - so 'Boruto'
        resolves to 'Boruto: Two Blue Vortex', while an input that could mean
        two different rows resolves to nothing and the caller reports it.
        """
        exact = self._find_exact(title)
        if exact is not None:
            return exact

        wanted = title.strip().lower()
        if len(wanted) < 3:
            # Too short to guess from safely.
            return None

        partial = [
            r
            for r in self._all_series()
            if wanted in r.title.strip().lower() or r.title.strip().lower() in wanted
        ]
        if len(partial) == 1:
            return partial[0]
        return None

    def _not_found(self, action: str, title: str) -> ActionResult:
        titles = [r.title for r in self._all_series()]
        hint = f" Your library has: {', '.join(titles[:8])}." if titles else " Your library is empty."
        return ActionResult(
            action, "error", f"'{title}' is not in your library.{hint}"
        )

    # -- dispatch ----------------------------------------------------------- #

    def execute_plan(
        self, actions: list, user_request: str, *, confirmed: bool = False
    ) -> ExecutionReport:
        report = ExecutionReport()
        for action in actions:
            if action.requires_confirmation and not confirmed:
                result = ActionResult(
                    action.action, "cancelled", f"Awaiting confirmation: {action.describe()}"
                )
            else:
                result = self.execute(action)
            report.results.append(result)
            record_action(
                self.session,
                user_id=self.user_id,
                user_request=user_request,
                action_name=action.action,
                payload=action.model_dump(mode="json"),
                status=result.status,
                result_summary=result.summary,
                confirmed=confirmed and action.requires_confirmation,
            )
        return report

    def execute(self, action) -> ActionResult:
        handler = getattr(self, f"_do_{action.action}", None)
        if handler is None:  # defence in depth; validator should prevent this
            return ActionResult(
                action.action, "rejected", f"No handler for action '{action.action}'."
            )
        try:
            return handler(action)
        except MangaAPIError as exc:
            return ActionResult(action.action, "error", str(exc))
        except Exception as exc:  # never let one action crash the app
            self.session.rollback()
            return ActionResult(
                action.action,
                "error",
                f"Unexpected failure in {action.action}: {type(exc).__name__}: {exc}",
            )

    # -- database domain ---------------------------------------------------- #

    def _do_add_series(self, action: AddSeries) -> ActionResult:
        if self._find_exact(action.title) is not None:
            return ActionResult(
                action.action, "error", f"'{action.title}' is already in your library."
            )
        row = Series(
            user_id=self.user_id,
            title=action.title,
            status=action.status,
            chapters_read=action.chapters_read,
            notes=action.notes,
        )
        self.session.add(row)
        self.session.flush()
        return ActionResult(
            action.action,
            "ok",
            f"Added '{row.title}' as {row.status.replace('_', ' ')}.",
            row.to_dict(),
        )

    def _do_update_progress(self, action: UpdateProgress) -> ActionResult:
        row = self._find(action.title)
        if row is None:
            return self._not_found(action.action, action.title)
        previous = row.chapters_read
        row.chapters_read = action.chapters_read
        # Reading past the last-known release means we're caught up.
        if row.latest_chapter is not None and action.chapters_read > row.latest_chapter:
            row.latest_chapter = action.chapters_read
        if row.status == "plan_to_read" and action.chapters_read > 0:
            row.status = "reading"
        row.updated_at = utcnow()
        self.session.flush()
        return ActionResult(
            action.action,
            "ok",
            f"'{row.title}': chapter {previous:g} to {row.chapters_read:g}.",
            row.to_dict(),
        )

    def _do_set_status(self, action: SetStatus) -> ActionResult:
        row = self._find(action.title)
        if row is None:
            return self._not_found(action.action, action.title)
        previous = row.status
        row.status = action.status
        row.updated_at = utcnow()
        self.session.flush()
        return ActionResult(
            action.action,
            "ok",
            f"'{row.title}': {previous.replace('_', ' ')} to "
            f"{row.status.replace('_', ' ')}.",
            row.to_dict(),
        )

    def _do_rate_series(self, action: RateSeries) -> ActionResult:
        row = self._find(action.title)
        if row is None:
            return self._not_found(action.action, action.title)
        row.rating = action.rating
        row.updated_at = utcnow()
        self.session.flush()
        return ActionResult(
            action.action, "ok", f"Rated '{row.title}' {action.rating:g}/10.", row.to_dict()
        )

    def _do_remove_series(self, action: RemoveSeries) -> ActionResult:
        row = self._find(action.title)
        if row is None:
            return self._not_found(action.action, action.title)
        title = row.title
        snapshot = row.to_dict()
        self.session.delete(row)
        self.session.flush()
        return ActionResult(
            action.action, "ok", f"Deleted '{title}' from your library.", snapshot
        )

    def _do_query_library(self, action: QueryLibrary) -> ActionResult:
        rows = self._all_series()
        if action.status:
            rows = [r for r in rows if r.status == action.status]
        if action.behind_only:
            rows = [r for r in rows if (r.chapters_behind or 0) > 0]
        if action.min_rating is not None:
            rows = [r for r in rows if (r.rating or 0) >= action.min_rating]

        if action.sort_by == "title":
            rows.sort(key=lambda r: r.title.lower())
        elif action.sort_by == "chapters_behind":
            rows.sort(key=lambda r: r.chapters_behind or 0, reverse=True)
        elif action.sort_by == "rating":
            rows.sort(key=lambda r: r.rating or -1, reverse=True)

        rows = rows[: action.limit]
        return ActionResult(
            action.action,
            "ok",
            f"Found {len(rows)} series.",
            [r.to_dict() for r in rows],
        )

    # -- API domain --------------------------------------------------------- #

    def _do_search_manga(self, action: SearchManga) -> ActionResult:
        results = self.api.search_manga(action.query, limit=action.limit)
        if not results:
            return ActionResult(
                action.action, "ok", f"No MangaDex results for '{action.query}'.", []
            )
        return ActionResult(
            action.action,
            "ok",
            f"{len(results)} MangaDex result(s) for '{action.query}'.",
            results,
        )

    def _do_check_new_chapters(self, action: CheckNewChapters) -> ActionResult:
        if action.title:
            row = self._find(action.title)
            if row is None:
                return self._not_found(action.action, action.title)
            targets = [row]
        else:
            targets = [r for r in self._all_series() if r.status == "reading"]
            if not targets:
                return ActionResult(
                    action.action,
                    "ok",
                    "Nothing is marked as currently reading, so there was nothing to check.",
                    [],
                )

        checked: list[dict] = []
        failures: list[str] = []
        for row in targets:
            try:
                if not row.mangadex_id:
                    match = self.api.resolve_series(row.title)
                    if match is None:
                        failures.append(f"{row.title} (not found on MangaDex)")
                        continue
                    row.mangadex_id = match["mangadex_id"]
                latest = self.api.latest_chapter(row.mangadex_id)
                if latest is None:
                    failures.append(f"{row.title} (no English chapters listed)")
                    continue
                row.latest_chapter = latest
                row.last_checked = utcnow()
                self.session.flush()
                checked.append(
                    {
                        "title": row.title,
                        "chapters_read": row.chapters_read,
                        "latest_chapter": latest,
                        "chapters_behind": row.chapters_behind,
                    }
                )
            except MangaAPIError as exc:
                failures.append(f"{row.title} ({exc})")

        behind = [c for c in checked if (c["chapters_behind"] or 0) > 0]
        parts = [f"Checked {len(checked)} series."]
        if behind:
            parts.append(
                "Behind on: "
                + ", ".join(f"{c['title']} (+{c['chapters_behind']:g})" for c in behind)
                + "."
            )
        elif checked:
            parts.append("You're caught up on everything checked.")
        if failures:
            parts.append("Could not check: " + "; ".join(failures) + ".")

        status = "ok" if checked or not failures else "error"
        return ActionResult(action.action, status, " ".join(parts), checked)

    def _do_get_recommendations(self, action: GetRecommendations) -> ActionResult:
        own_titles = {r.title for r in self._all_series()}
        results = self.api.recommendations(
            genres=action.genres,
            similar_to=action.similar_to,
            finished_only=action.finished_only,
            max_chapters=action.max_chapters,
            limit=action.limit,
            exclude_titles=own_titles,
        )
        if not results:
            return ActionResult(
                action.action,
                "ok",
                "No recommendations matched those filters. Try loosening them.",
                [],
            )
        return ActionResult(
            action.action, "ok", f"{len(results)} recommendation(s).", results
        )

    # -- file domain -------------------------------------------------------- #

    EXPORT_COLUMNS = (
        "title",
        "status",
        "chapters_read",
        "rating",
        "latest_chapter",
        "chapters_behind",
        "notes",
    )

    def _do_export_list(self, action: ExportList) -> ActionResult:
        rows = self._all_series()
        if action.status:
            rows = [r for r in rows if r.status == action.status]
        if not rows:
            return ActionResult(
                action.action, "error", "There is nothing to export - your library is empty."
            )

        records = [r.to_dict() for r in rows]
        if action.format == "csv":
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer, fieldnames=list(self.EXPORT_COLUMNS), extrasaction="ignore"
            )
            writer.writeheader()
            for record in records:
                writer.writerow({k: record.get(k, "") for k in self.EXPORT_COLUMNS})
            content, mime, ext = buffer.getvalue(), "text/csv", "csv"
        elif action.format == "json":
            content = json.dumps(records, indent=2)
            mime, ext = "application/json", "json"
        else:
            header = "| Title | Status | Read | Latest | Rating |\n|---|---|---|---|---|\n"
            lines = [
                f"| {r['title']} | {r['status'].replace('_', ' ')} | "
                f"{r['chapters_read']:g} | "
                f"{r['latest_chapter'] if r['latest_chapter'] is not None else '-'} | "
                f"{r['rating'] if r['rating'] is not None else '-'} |"
                for r in records
            ]
            content = "# My manga list\n\n" + header + "\n".join(lines) + "\n"
            mime, ext = "text/markdown", "md"

        scope = f"-{action.status}" if action.status else ""
        return ActionResult(
            action.action,
            "ok",
            f"Exported {len(records)} series as {action.format}.",
            {"count": len(records)},
            download={
                "filename": f"manga-list{scope}.{ext}",
                "content": content,
                "mime": mime,
            },
        )

    def _do_import_list(self, action: ImportList) -> ActionResult:
        text = self.upload_provider()
        if not text:
            return ActionResult(
                action.action,
                "error",
                "No CSV has been uploaded. Use the 'Import CSV' uploader in the "
                "sidebar first, then ask again.",
            )
        try:
            reader = csv.DictReader(io.StringIO(text))
            incoming = list(reader)
        except csv.Error as exc:
            return ActionResult(action.action, "error", f"Could not parse the CSV: {exc}")

        if not incoming or "title" not in (reader.fieldnames or []):
            return ActionResult(
                action.action,
                "error",
                "The CSV needs at least a 'title' column with one row per series.",
            )

        added = updated = skipped = 0
        problems: list[str] = []
        for line_no, record in enumerate(incoming, start=2):
            title = (record.get("title") or "").strip()
            if not title:
                problems.append(f"row {line_no}: blank title")
                continue

            status = (record.get("status") or "plan_to_read").strip().lower()
            status = status.replace(" ", "_").replace("-", "_")
            if status not in VALID_STATUSES:
                status = "plan_to_read"

            try:
                chapters = float(record.get("chapters_read") or 0)
                chapters = max(0.0, min(chapters, 10_000.0))
            except (TypeError, ValueError):
                chapters = 0.0

            rating_raw = (record.get("rating") or "").strip()
            rating: float | None = None
            if rating_raw:
                try:
                    rating = max(0.0, min(float(rating_raw), 10.0))
                except (TypeError, ValueError):
                    rating = None

            existing = self._find_exact(title)
            if existing is not None:
                if not action.overwrite:
                    skipped += 1
                    continue
                existing.status = status
                existing.chapters_read = chapters
                if rating is not None:
                    existing.rating = rating
                existing.updated_at = utcnow()
                updated += 1
            else:
                self.session.add(
                    Series(
                        user_id=self.user_id,
                        title=title,
                        status=status,
                        chapters_read=chapters,
                        rating=rating,
                    )
                )
                added += 1
            self.session.flush()

        summary = f"Imported {added} new, updated {updated}, skipped {skipped}."
        if problems:
            summary += f" Issues: {'; '.join(problems[:5])}."
        return ActionResult(
            action.action,
            "ok",
            summary,
            {"added": added, "updated": updated, "skipped": skipped},
        )
