(function () {
    "use strict";

    /* -------------------------------------------------------------------
       Prediction form: 4-step wizard + live profile-strength preview
       ------------------------------------------------------------------- */
    var form = document.getElementById("predict-form");

    if (form) {
        var steps = Array.prototype.slice.call(form.querySelectorAll(".step"));
        var labels = Array.prototype.slice.call(form.querySelectorAll("[data-step-label]"));
        var nav = form.querySelector(".form-nav");
        var btnPrev = form.querySelector("[data-prev]");
        var btnNext = form.querySelector("[data-next]");
        var current = 0;

        function showStep(index, focusFirst) {
            current = Math.max(0, Math.min(steps.length - 1, index));

            steps.forEach(function (step, i) {
                step.classList.toggle("is-active", i === current);
            });
            labels.forEach(function (label, i) {
                label.classList.toggle("is-active", i === current);
                label.classList.toggle("is-done", i < current);
                if (i === current) { label.setAttribute("aria-current", "step"); }
                else { label.removeAttribute("aria-current"); }
            });

            nav.classList.toggle("is-first", current === 0);
            nav.classList.toggle("is-last", current === steps.length - 1);

            if (focusFirst) {
                var first = steps[current].querySelector("input, select");
                if (first) { first.focus({ preventScroll: true }); }
                form.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        }

        function stepIsValid(step) {
            var fields = step.querySelectorAll("input, select");
            for (var i = 0; i < fields.length; i++) {
                if (!fields[i].checkValidity()) {
                    fields[i].reportValidity();
                    return false;
                }
            }
            return true;
        }

        btnNext.addEventListener("click", function () {
            if (stepIsValid(steps[current])) { showStep(current + 1, true); }
        });
        btnPrev.addEventListener("click", function () { showStep(current - 1, true); });

        // Enter inside a number box moves forward instead of submitting early.
        form.addEventListener("keydown", function (event) {
            if (event.key === "Enter" && event.target.tagName === "INPUT" && current < steps.length - 1) {
                event.preventDefault();
                btnNext.click();
            }
        });

        form.addEventListener("submit", function (event) {
            // Find the first invalid field across all steps (hidden steps can't
            // show validation bubbles, so switch to that step first).
            for (var i = 0; i < steps.length; i++) {
                var bad = steps[i].querySelector(":invalid");
                if (bad) {
                    event.preventDefault();
                    showStep(i, false);
                    bad.reportValidity();
                    return;
                }
            }
            var submit = form.querySelector("[data-submit]");
            submit.classList.add("is-loading");
            submit.textContent = "Calculating...";
        });

        // Jump to the first step that has a server-side error.
        var firstError = form.querySelector(".has-error");
        var startAt = 0;
        if (firstError) {
            startAt = steps.indexOf(firstError.closest(".step"));
            if (startAt < 0) { startAt = 0; }
        }
        showStep(startAt, false);

        /* ---- live meter (same formulas as the server) ---- */
        var overallEl = document.getElementById("live-overall");
        var cap = parseFloat(overallEl.getAttribute("data-cap")) || 60;

        function num(name) {
            var el = form.elements[name];
            var value = el ? parseFloat(el.value) : 0;
            return isNaN(value) ? 0 : value;
        }
        function clamp(v) { return Math.max(0, Math.min(100, v)); }

        function setMeter(key, value) {
            var rounded = Math.round(value);
            document.getElementById("v-" + key).textContent = rounded;
            document.getElementById("m-" + key).style.setProperty("--v", clamp(rounded));
        }

        function updateLive() {
            var academic = num("cgpa") * 10;
            var technical =
                num("coding_skill_score") * 0.30 +
                num("logical_reasoning_score") * 0.20 +
                num("aptitude_score") * 0.20 +
                num("mock_interview_score") * 0.20 +
                num("communication_skill_score") * 0.10;
            var experienceRaw =
                num("internships_count") * 5 +
                num("projects_count") * 3 +
                num("certifications_count") * 2 +
                num("hackathons_participated") * 2 +
                num("github_repos");
            var experience = Math.min(100, experienceRaw / cap * 100);
            var professional =
                num("leadership_score") * 0.35 +
                num("communication_skill_score") * 0.35 +
                num("extracurricular_score") * 0.30;

            var overall = clamp(
                academic * 0.2 + technical * 0.4 + experience * 0.2 + professional * 0.2 -
                num("backlogs") * 5
            );

            setMeter("academic", clamp(academic));
            setMeter("technical", technical);
            setMeter("experience", experience);
            setMeter("professional", professional);
            overallEl.textContent = Math.round(overall);
        }

        // Slider read-outs
        form.querySelectorAll('input[type="range"]').forEach(function (slider) {
            var out = form.querySelector('[data-out="' + slider.name + '"]');
            function sync() { if (out) { out.textContent = slider.value; } }
            slider.addEventListener("input", sync);
            sync();
        });

        form.addEventListener("input", updateLive);
        updateLive();
    }

    /* -------------------------------------------------------------------
       Resume upload: show file name, support drag and drop
       ------------------------------------------------------------------- */
    var resumeForm = document.getElementById("resume-form");

    if (resumeForm) {
        var fileInput = document.getElementById("resume-file");
        var title = document.getElementById("drop-title");

        function showName() {
            if (fileInput.files && fileInput.files.length) {
                title.textContent = fileInput.files[0].name;
            }
        }
        fileInput.addEventListener("change", showName);

        ["dragenter", "dragover"].forEach(function (name) {
            resumeForm.addEventListener(name, function (event) {
                event.preventDefault();
                resumeForm.classList.add("is-over");
            });
        });
        ["dragleave", "drop"].forEach(function (name) {
            resumeForm.addEventListener(name, function (event) {
                event.preventDefault();
                resumeForm.classList.remove("is-over");
            });
        });
        resumeForm.addEventListener("drop", function (event) {
            if (event.dataTransfer && event.dataTransfer.files.length) {
                fileInput.files = event.dataTransfer.files;
                showName();
            }
        });

        resumeForm.addEventListener("submit", function () {
            var button = resumeForm.querySelector("button[type=submit]");
            button.classList.add("is-loading");
            button.textContent = "Analysing...";
        });
    }

    /* -------------------------------------------------------------------
       Print button on result pages
       ------------------------------------------------------------------- */
    document.querySelectorAll("[data-print]").forEach(function (button) {
        button.addEventListener("click", function () { window.print(); });
    });
})();

/* -------------------------------------------------------------------
   Skill assessment: block submission until every question is answered
   ------------------------------------------------------------------- */
(function () {
    "use strict";
    var quizForm = document.getElementById("assessment-form");
    if (!quizForm) { return; }

    var errorBox = document.getElementById("quiz-error");

    quizForm.addEventListener("submit", function (event) {
        var questions = quizForm.querySelectorAll("[data-question]");
        var firstUnanswered = null;

        questions.forEach(function (fieldset) {
            var answered = fieldset.querySelector('input[type="radio"]:checked');
            if (!answered && !firstUnanswered) { firstUnanswered = fieldset; }
        });

        if (firstUnanswered) {
            event.preventDefault();
            if (errorBox) { errorBox.hidden = false; }
            firstUnanswered.scrollIntoView({ behavior: "smooth", block: "center" });
        } else if (errorBox) {
            errorBox.hidden = true;
        }
    });
})();
