"""An evidence-driven topic frontier, with no campaign round allowance."""

from __future__ import annotations

from ..contracts.base import canonical_digest


class TopicResearch:
    """Explore every initial topic, then continue branches making measured progress.

    A failed refinement returns for diagnosis; it does not end the topic.
    New topics may be proposed from measured failures. Transport failures remain unresolved
    failures; they never count as convergence or evidence against a hypothesis.
    """

    def __init__(self, pods):
        self.pods = {pod: {"planned": False, "topics": {}, "blocked": None,
            "unresolved_topics": {}} for pod in pods}

    def current(self, pod):
        state = self.pods[pod]
        if state["blocked"]:
            return None
        if not state["planned"]:
            return {"phase": "plan", "topic": None}
        topics = list(state["topics"].values())
        # Breadth precedes tuning: every untouched initial topic gets its test.
        candidates = sorted([t for t in topics if t["status"] == "unexplored"], key=lambda t: t["visits"])
        candidates += sorted([t for t in topics if t["status"] in {"refine", "review"}], key=lambda t: t["visits"])
        if not candidates:
            if state.get("unresolved_topics"):
                state["blocked"] = {"status": "unresolved_topics",
                    "topics": state["unresolved_topics"]}
            return None
        topic = candidates[0]
        return {"phase": "explore" if topic["status"] == "unexplored" else topic["status"],
            "topic": topic}

    def add_topics(self, pod, topics):
        state = self.pods[pod]
        existing = {t["mechanism_digest"] for t in state["topics"].values()}
        added = 0
        for topic in topics:
            digest = canonical_digest({"mechanism": " ".join(topic.mechanism.lower().split())})
            if topic.name in state["topics"] or digest in existing:
                continue
            state["topics"][topic.name] = {
                **topic.model_dump(mode="json"), "mechanism_digest": digest,
                "status": "unexplored", "evaluations": 0, "best_score": None,
                "visits": 0,
                "best_strategy": None, "search_signatures": [], "stop_reason": None}
            existing.add(digest)
            added += 1
        return added

    def _unresolved_topic(self, state, topic, status, reason):
        """Preserve failed evaluation work without cancelling independent topics."""
        failure = {"status": status, "reason": reason}
        state.setdefault("unresolved_topics", {})[topic["name"]] = failure
        topic.update(status="unresolved", stop_reason=reason)

    def observe(self, pod, context, result, candidates, evidence_signature):
        state = self.pods[pod]
        status = result.get("status")
        if status in {"provider_failed", "invalid_batch"}:
            state["blocked"] = {"status": status, "reason": result.get("reason", result.get("error", status))}
            return
        if context["phase"] == "plan":
            if status != "topics_planned":
                state["blocked"] = {"status": "missing_initial_topic_plan"}
            else:
                state["planned"] = True
            return
        topic = state["topics"][context["topic"]["name"]]
        topic["visits"] += 1
        if status == "evidence_requested":
            request = result.get("research_request_signature", evidence_signature)
            if request in topic["search_signatures"] and evidence_signature == "no_new_evidence":
                self._unresolved_topic(state, topic, "repeated_retrieval_without_evidence",
                    "Repeated retrieval added no evidence; the economic hypothesis remains untested or unresolved")
            else:
                if request not in topic["search_signatures"]:
                    topic["search_signatures"].append(request)
                if evidence_signature == "no_new_evidence":
                    topic.update(status="review", last_outcome="retrieval_added_no_new_evidence")
            return
        if status == "retained":
            if topic["evaluations"] == 0 and topic.get("last_outcome") != "retrieval_added_no_new_evidence":
                state["blocked"] = {"status": "topic_retained_before_testing", "topic": topic["name"]}
            else:
                topic.update(status="exhausted", stop_reason=result.get("diagnosis", "agent_reports_no_further_measured_lead"))
            return
        if not candidates:
            self._unresolved_topic(state, topic, status or "no_evaluated_proposal",
                "No new proposal could be evaluated after feedback; this is not evidence of no economic edge")
            return
        previous = topic["best_score"]
        best = None
        for identity, review in candidates:
            score = review.get("selection_score")
            if score is None:
                continue
            # Compare the same predictive formula for parent and child, rather
            # than rewarding a prettier realized portfolio return.
            comparison = review.get("parent_comparison")
            if comparison is not None:
                score, baseline = comparison["candidate_score"], comparison["parent_score"]
                if score is None or baseline is None or score <= baseline + 1e-10:
                    continue
            if best is None or score > best[0]:
                best = (score, identity)
        topic["evaluations"] += len(candidates)
        compared = any("parent_comparison" in review for _, review in candidates)
        if best is not None and (compared or previous is None or best[0] > previous + 1e-10):
            topic.update(best_score=best[0], best_strategy=best[1], status="refine")
        else:
            topic.update(status="review", last_outcome="no_improvement_in_predictive_selection_score")

    def snapshot(self):
        return {"stopping_rule": "topic_frontier_convergence_or_explicit_unresolved_failure",
            "selection_method": "predictive_score_v1",
            "pods": self.pods}
