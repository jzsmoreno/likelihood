'use strict';

/*
 * Model Training workspace.
 *
 * Every view is driven by the real API: the model picker comes from
 * /api/training/models, the job list from /api/training/jobs (polled while a job
 * is running) and the job detail — with its log, timing and artifact — from
 * /api/training/jobs/<id>.  There is no synthetic data: a job that has not
 * finished simply shows its current status until it does.
 */
(function () {
  var API = '/api';
  var state = {
    health: null,
    models: [],
    jobs: [],
    jobsLoaded: false,
    artifactCount: null,
    job: null,
    pollTimer: 0,
    // create wizard
    wStep: 1,
    wKind: 'fit',
    wModel: '',
    wMethod: '',
    wName: '',
    wTimeout: '',
    wParams: '{}',
    wArgs: '{}',
    wArtifact: '',
  };

  var dom = {};
  var confirmCallback = null;

  function byId(id) { return document.getElementById(id); }

  function esc(value) {
    var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' };
    return String(value === null || value === undefined ? '' : value)
      .replace(/[&<>"']/g, function (ch) { return map[ch]; });
  }

  function each(list, callback) { Array.prototype.forEach.call(list || [], callback); }

  function api(path, options) {
    var config = options || {};
    var headers = { Accept: 'application/json' };
    if (config.body !== undefined) headers['Content-Type'] = 'application/json';
    return fetch(path, {
      method: config.method || 'GET',
      headers: headers,
      body: config.body === undefined ? undefined : JSON.stringify(config.body),
      credentials: 'same-origin',
    }).then(function (response) {
      return response.text().then(function (text) {
        var data = null;
        try { data = text ? JSON.parse(text) : {}; } catch (error) { data = { error: text }; }
        if (!response.ok || (data && data.success === false)) {
          var failure = new Error((data && data.error) || ('HTTP ' + response.status));
          failure.status = response.status;
          failure.data = data || {};
          throw failure;
        }
        return data;
      });
    });
  }

  function toast(message, isError) {
    if (!dom.toast) return;
    dom.toast.textContent = message;
    dom.toast.classList.toggle('is-error', Boolean(isError));
    dom.toast.classList.add('is-visible');
    window.clearTimeout(toast.timer);
    toast.timer = window.setTimeout(function () {
      dom.toast.classList.remove('is-visible');
    }, 2800);
  }
  toast.timer = 0;

  function fmt(value) { return window.LMCharts ? LMCharts.fmtNumber(value) : String(value); }
  function fmtMs(value) { return window.LMCharts ? LMCharts.fmtMs(value) : (value + ' ms'); }
  function sec(value) {
    if (value === null || value === undefined) return '—';
    return window.LMCharts ? (LMCharts.fmtNumber(value, 3) + ' s') : value + ' s';
  }
  function isoDate(value) {
    if (!value) return '—';
    var date = new Date(value);
    return isNaN(date.getTime()) ? String(value) : date.toLocaleString('es-ES');
  }

  function statusBadge(status) {
    var label = {
      pending: 'Pendiente', running: 'En ejecución', succeeded: 'Completado',
      failed: 'Fallido', cancelled: 'Cancelado',
    }[status] || status;
    return '<span class="status-badge status-' + esc(status) + '"><span class="dot"></span>' + esc(label) + '</span>';
  }

  function kindBadge(kind) {
    return '<span class="kind-badge kind-' + esc(kind) + '">' + esc(kind) + '</span>';
  }

  function emptyState(icon, title, text, action) {
    return '<div class="empty-state"><span class="empty-icon" aria-hidden="true">' + icon + '</span>' +
      '<h3>' + esc(title) + '</h3><p>' + esc(text) + '</p>' + (action || '') + '</div>';
  }

  /* ------------------------------------------------------------------ */
  /* Shell: sidebar, theme, service status and routing                   */
  /* ------------------------------------------------------------------ */

  function isMobileLayout() { return window.matchMedia('(max-width: 900px)').matches; }

  function syncSidebarToggle() {
    var toggle = byId('sidebar-toggle');
    var overlay = byId('sidebar-overlay');
    var mobile = isMobileLayout();
    var expanded = mobile
      ? document.body.classList.contains('sidebar-open')
      : !document.body.classList.contains('sidebar-collapsed');
    if (toggle) toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    if (overlay) overlay.hidden = !(mobile && document.body.classList.contains('sidebar-open'));
  }

  function closeMobileSidebar() { document.body.classList.remove('sidebar-open'); syncSidebarToggle(); }

  function bindSidebar() {
    var toggle = byId('sidebar-toggle');
    var overlay = byId('sidebar-overlay');
    if (toggle) {
      toggle.addEventListener('click', function () {
        if (isMobileLayout()) document.body.classList.toggle('sidebar-open');
        else document.body.classList.toggle('sidebar-collapsed');
        syncSidebarToggle();
      });
    }
    if (overlay) overlay.addEventListener('click', closeMobileSidebar);
    window.addEventListener('resize', function () {
      if (!isMobileLayout()) closeMobileSidebar();
      syncSidebarToggle();
    });
    syncSidebarToggle();
  }

  function currentTheme() {
    var explicit = document.documentElement.getAttribute('data-theme');
    if (explicit === 'light' || explicit === 'dark') return explicit;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function bindTheme() {
    var toggle = byId('theme-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { window.localStorage.setItem('lm-theme', next); } catch (error) { /* optional */ }
      toast(next === 'dark' ? 'Tema oscuro activado' : 'Tema claro activado');
    });
  }

  function loadHealth() {
    return api('/api/health').then(function (data) {
      state.health = data;
      if (dom.serverStatus) {
        dom.serverStatus.classList.remove('is-ok', 'is-warn', 'is-error');
        dom.serverStatus.classList.add('is-ok');
        var dot = dom.serverStatus.querySelector('.dot');
        if (dot) dot.className = 'dot dot--ok';
      }
      if (dom.serverStatusText) dom.serverStatusText.textContent = 'En línea';
      if (dom.sideVersion) dom.sideVersion.textContent = 'v' + (data.version || '—');
      return data;
    }).catch(function () {
      if (dom.serverStatus) {
        dom.serverStatus.classList.remove('is-ok');
        dom.serverStatus.classList.add('is-error');
        var dot = dom.serverStatus.querySelector('.dot');
        if (dot) dot.className = 'dot dot--off';
      }
      if (dom.serverStatusText) dom.serverStatusText.textContent = 'Sin conexión';
    });
  }

  function route() {
    var match = window.location.pathname.match(/^\/training\/([^/]+)\/?$/);
    if (match && match[1]) showDetail(decodeURIComponent(match[1]));
    else showList();
  }

  function showList() {
    stopPolling();
    state.job = null;
    if (dom.listView) dom.listView.hidden = false;
    if (dom.detailView) dom.detailView.hidden = true;
    if (dom.crumbCurrent) dom.crumbCurrent.textContent = 'Jobs';
    loadJobs();
    loadHealth();
    loadArtifactCount();
  }

  function showDetail(jobId) {
    stopPolling();
    if (dom.listView) dom.listView.hidden = true;
    if (dom.detailView) dom.detailView.hidden = false;
    if (dom.crumbCurrent) dom.crumbCurrent.textContent = jobId;
    loadJob(jobId);
    loadHealth();
  }

  function stopPolling() {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = 0;
  }

  /* ------------------------------------------------------------------ */
  /* Jobs list                                                           */
  /* ------------------------------------------------------------------ */

  function loadJobs() {
    if (!dom.results) return;
    dom.results.setAttribute('aria-busy', 'true');
    return api('/api/training/jobs').then(function (data) {
      state.jobs = data.jobs || [];
      state.jobsLoaded = true;
      renderKpis();
      renderJobs();
      if (dom.countJobs) dom.countJobs.textContent = String(state.jobs.length);
    }).catch(function (error) {
      state.jobsLoaded = true;
      if (dom.kpis) dom.kpis.innerHTML = '';
      dom.results.innerHTML = emptyState(
        '⚠', 'No se pudieron cargar los jobs', error.message,
        '<button class="btn btn-secondary btn-sm" type="button" id="retry-jobs">Reintentar</button>'
      );
      if (dom.countJobs) dom.countJobs.textContent = '—';
    }).then(function () {
      dom.results.setAttribute('aria-busy', 'false');
    });
  }

  function loadArtifactCount() {
    if (!dom.countArtifacts) return;
    return api('/api/training/artifacts').then(function (data) {
      state.artifactCount = data.count || 0;
      if (dom.countArtifacts) dom.countArtifacts.textContent = String(state.artifactCount);
    }).catch(function () {
      if (dom.countArtifacts) dom.countArtifacts.textContent = '—';
    });
  }

  function renderKpis() {
    if (!dom.kpis) return;
    var counts = { total: 0, active: 0, succeeded: 0, failed: 0, artifacts: state.artifactCount || 0 };
    state.jobs.forEach(function (job) {
      counts.total += 1;
      if (job.status === 'pending' || job.status === 'running') counts.active += 1;
      if (job.status === 'succeeded') counts.succeeded += 1;
      if (job.status === 'failed') counts.failed += 1;
    });
    dom.kpis.innerHTML = [
      kpi('⇣', counts.total, 'Jobs', 'blue'),
      kpi('◔', counts.active, 'En curso', 'amber', counts.active ? 'en cola o ejecutando' : ''),
      kpi('✓', counts.succeeded, 'Completados', 'green'),
      kpi('⚠', counts.failed, 'Fallidos', 'lava'),
      kpi('{ }', counts.artifacts, 'Artefactos', '', 'guardados en disk')
    ].join('');
  }

  function kpi(icon, value, label, tone, note) {
    var color = { blue: 'var(--db-blue)', amber: 'var(--db-amber)', green: 'var(--db-green)', lava: 'var(--db-lava)' }[tone] || 'var(--db-ink)';
    return '<div class="kpi"><span class="kpi-icon" style="color:' + color + ';" aria-hidden="true">' +
      icon + '</span><div class="kpi-body"><div class="kpi-value">' + esc(value) + '</div>' +
      '<div class="kpi-label">' + esc(label) + '</div>' +
      (note ? '<div class="kpi-delta">' + esc(note) + '</div>' : '') + '</div></div>';
  }

  function filteredJobs() {
    var term = (dom.search ? dom.search.value : '').trim().toLowerCase();
    var kind = dom.kindFilter ? dom.kindFilter.value : '';
    var status = dom.statusFilter ? dom.statusFilter.value : '';
    return state.jobs.filter(function (job) {
      if (kind && job.kind !== kind) return false;
      if (status && job.status !== status) return false;
      if (!term) return true;
      return [job.name, job.job_id, job.model_id, job.model_name, job.method]
        .join(' ').toLowerCase().indexOf(term) !== -1;
    });
  }

  function renderJobs() {
    if (!dom.results) return;
    var rows = filteredJobs();
    if (!state.jobs.length) {
      dom.results.innerHTML = emptyState(
        '◔', 'Todavía no hay jobs',
        'Lanza un job de fit o evaluate para entrenar un método del catálogo y conservar el resultado como artefacto.',
        '<button class="btn btn-primary btn-sm" type="button" id="new-job-from-empty">＋ Crear job</button>'
      );
    } else if (!rows.length) {
      dom.results.innerHTML = emptyState(
        '⌕', 'Ningún job coincide', 'Ajusta la búsqueda o los filtros para ver otros jobs.',
        '<button class="btn btn-secondary btn-sm" type="button" id="clear-filters">Restablecer filtros</button>'
      );
    } else {
      dom.results.innerHTML = '<div class="table-wrap"><table class="table">' +
        '<caption class="sr-only">Jobs de entrenamiento</caption><thead><tr>' +
        '<th>Job</th><th>Modelo</th><th>Método</th><th>Tipo</th><th>Estado</th>' +
        '<th class="num-head">Duración</th><th class="num-head">Creado</th><th class="cell-actions">Acciones</th></tr></thead><tbody>' +
        rows.map(jobRow).join('') + '</tbody></table></div>';
    }
    if (dom.resultBar) dom.resultBar.hidden = !state.jobs.length;
    if (dom.resultCount) {
      dom.resultCount.textContent = rows.length + ' de ' + state.jobs.length +
        (state.jobs.length === 1 ? ' job' : ' jobs');
    }
    if (dom.resultHint) {
      var active = state.jobs.filter(function (item) {
        return item.status === 'pending' || item.status === 'running';
      }).length;
      dom.resultHint.textContent = active ? active + ' en curso' : 'sin jobs activos';
    }
    wireRowActions();
  }

  function jobRow(job) {
    var duration = job.elapsed === null || job.elapsed === undefined ? '—' : sec(job.elapsed);
    var artifact = job.artifact_id ? esc(job.artifact_id) : '—';
    return '<tr>' +
      '<td class="cell-main"><a class="cell-link" href="/training/' + esc(job.job_id) + '">' +
        (job.name ? esc(job.name) : esc(job.job_id)) + '</a>' +
        '<span class="cell-sub mono">' + esc(job.job_id) + '</span></td>' +
      '<td><div class="cell-main">' + esc(job.model_name) + '</div>' +
        '<span class="cell-sub mono">' + esc(job.model_id) + '</span></td>' +
      '<td><code>' + esc(job.method) + '</code></td>' +
      '<td>' + kindBadge(job.kind) + '</td>' +
      '<td>' + statusBadge(job.status) + '</td>' +
      '<td class="num">' + duration + '</td>' +
      '<td class="num">' + esc(isoDate(job.created_at)) + '</td>' +
      '<td class="cell-actions">' +
        '<a class="btn btn-ghost btn-sm" href="/training/' + esc(job.job_id) + '">Abrir</a>' +
        (job.status === 'pending'
          ? '<button class="btn btn-secondary btn-sm" type="button" data-action="cancel" data-id="' + esc(job.job_id) + '">Cancelar</button>'
          : '') +
        '<button class="btn btn-ghost btn-sm btn-danger" type="button" data-action="delete" data-id="' + esc(job.job_id) + '" title="Eliminar job">✕</button>' +
      '</td></tr>';
  }

  function wireRowActions() {
    if (!dom.results) return;
    each(dom.results.querySelectorAll('[data-action="open"]'), function (button) {
      button.addEventListener('click', function () {
        window.location.href = '/training/' + button.getAttribute('data-id');
      });
    });
    each(dom.results.querySelectorAll('[data-action="cancel"]'), function (button) {
      button.addEventListener('click', function (event) {
        event.preventDefault();
        confirmCallback = function () { return cancelJob(button.getAttribute('data-id')); };
        openConfirm('Cancelar job', 'El job en cola se cancelará y no llegará a ejecutarse.');
      });
    });
    each(dom.results.querySelectorAll('[data-action="delete"]'), function (button) {
      button.addEventListener('click', function (event) {
        event.preventDefault();
        var id = button.getAttribute('data-id');
        confirmCallback = function () { return deleteJob(id); };
        openConfirm('Eliminar job', 'El job y (si es fit) su artefacto se eliminarán del servidor.');
      });
    });
    var retry = dom.results.querySelector('#retry-jobs');
    if (retry) retry.addEventListener('click', loadJobs);
    var newEmpty = dom.results.querySelector('#new-job-from-empty');
    if (newEmpty) newEmpty.addEventListener('click', openWizard);
    var clearFilters = dom.results.querySelector('#clear-filters');
    if (clearFilters) clearFilters.addEventListener('click', function () {
      if (dom.search) dom.search.value = '';
      if (dom.kindFilter) dom.kindFilter.value = '';
      if (dom.statusFilter) dom.statusFilter.value = '';
      renderJobs();
    });
  }

  /* ------------------------------------------------------------------ */
  /* Job detail                                                          */
  /* ------------------------------------------------------------------ */

  function loadJob(jobId) {
    if (!dom.detail) return;
    dom.detail.innerHTML = '<div class="loader"><span class="spinner"></span> Cargando job…</div>';
    return api('/api/training/jobs/' + encodeURIComponent(jobId)).then(function (data) {
      state.job = data.job || data;
      renderJobDetail(state.job);
    }).catch(function (error) {
      state.job = null;
      dom.detail.innerHTML = emptyState(
        '⚠', 'Job no encontrado', error.message,
        '<a class="btn btn-secondary btn-sm" href="/training">Volver a la lista</a>'
      );
    });
  }

  function renderJobDetail(job) {
    if (!dom.detail) return;
    var isFit = job.kind === 'fit';
    var artifact = job.artifact || null;
    var model = job.model || null;
    var instance = job.instance || {};
    var paramsText = JSON.stringify(job.parameters || {}, null, 2);
    var argsText = JSON.stringify(job.arguments || {}, null, 2);

    var header =
      '<div class="job-detail-header">' +
        '<div class="job-header-main">' +
          '<div class="page-eyebrow">Training · job</div>' +
          '<h1 class="job-title"><span class="job-title-icon" aria-hidden="true">◔</span> ' +
            (job.name ? esc(job.name) : esc(job.job_id)) + '</h1>' +
          '<div class="job-id-line mono">' + esc(job.job_id) + '</div>' +
          '<div class="job-meta-row">' + kindBadge(job.kind) + ' ' + statusBadge(job.status) + '</div>' +
        '</div>' +
        '<div class="job-header-actions">' +
          (job.status === 'pending'
            ? '<button class="btn btn-secondary btn-sm" type="button" data-action="detail-cancel">Cancelar</button>' : '') +
          '<button class="btn btn-ghost btn-sm btn-danger" type="button" data-action="detail-delete">Eliminar job</button>' +
        '</div>' +
      '</div>';

    var metaGrid =
      '<section class="meta-grid" aria-label="Detalles del job">' +
      '<div class="meta-item"><span class="meta-label">Modelo</span><span class="meta-value">' +
        (model ? esc(model.name) : esc(job.model_name)) + '</span><span class="meta-sub mono">' + esc(job.model_id) + '</span></div>' +
      '<div class="meta-item"><span class="meta-label">Método</span><code>' + esc(job.method) + '</code></div>' +
      '<div class="meta-item"><span class="meta-label">Creado</span><span class="meta-value">' + esc(isoDate(job.created_at)) + '</span></div>' +
      '<div class="meta-item"><span class="meta-label">Iniciado</span><span class="meta-value">' + esc(isoDate(job.started_at)) + '</span></div>' +
      '<div class="meta-item"><span class="meta-label">Finalizado</span><span class="meta-value">' + esc(isoDate(job.finished_at)) + '</span></div>' +
      '<div class="meta-item"><span class="meta-label">Duración total</span><span class="meta-value">' + sec(job.elapsed) + '</span></div>' +
      '<div class="meta-item"><span class="meta-label">Cómputo (sandbox)</span><span class="meta-value">' + sec(job.compute_seconds) + '</span></div>' +
      '<div class="meta-item"><span class="meta-label">Timeout</span><span class="meta-value">' + (job.timeout_seconds ? esc(job.timeout_seconds) + ' s' : '—') + '</span></div>' +
      '<div class="meta-item"><span class="meta-label">Intento</span><span class="meta-value">' + esc(job.attempts || 0) + '</span></div>' +
      '</section>';

    var errorBox = '';
    if (job.error) {
      errorBox =
        '<section class="panel panel-error" aria-label="Error del job">' +
          '<div class="panel-header"><span class="panel-icon" aria-hidden="true">⚠</span>' +
          '<h2 class="panel-title">El job falló</h2></div>' +
          '<div class="panel-body"><pre class="error-pre">' + esc(job.error) + '</pre></div>' +
        '</section>';
    }

    var resultPanel = '';
    if (job.status === 'succeeded') {
      resultPanel =
        '<section class="panel" aria-label="Resultado">' +
          '<div class="panel-header"><span class="panel-icon" aria-hidden="true">✓</span>' +
          '<h2 class="panel-title">Resultado</h2></div>' +
          '<div class="panel-body">' +
            '<div class="meta-grid">' +
              '<div class="meta-item"><span class="meta-label">Tipo de resultado</span><code>' + esc(job.result_type || '—') + '</code></div>' +
              '<div class="meta-item"><span class="meta-label">Representación</span><code>' + esc(job.result_repr || '—') + '</code></div>' +
              '<div class="meta-item"><span class="meta-label">Instancia</span><code>' + esc(instance.type || '—') + '</code></div>' +
            '</div>' +
            '<details class="details-block"><summary>Parámetros del constructor (JSON)</summary>' +
              '<pre class="code-block">' + esc(paramsText) + '</pre></details>' +
            '<details class="details-block"><summary>Argumentos del método (JSON)</summary>' +
              '<pre class="code-block">' + esc(argsText) + '</pre></details>' +
          '</div>' +
        '</section>';
    }

    var artifactPanel = '';
    if (artifact) {
      var exists = artifact.exists === true;
      artifactPanel =
        '<section class="panel artifact-panel" aria-label="Artefacto">' +
          '<div class="panel-header">' +
            '<h2 class="panel-title"><span class="panel-icon" aria-hidden="true">{ }</span> Artefacto</h2>' +
            (exists
              ? '<div class="panel-actions"><button class="btn btn-secondary btn-sm" type="button" data-action="delete-artifact">Borrar artefacto</button></div>'
              : '') +
          '</div>' +
          '<div class="panel-body">' +
            '<div class="meta-grid artifact-meta">' +
              '<div class="meta-item"><span class="meta-label">ID</span><code>' + esc(artifact.artifact_id) + '</code></div>' +
              '<div class="meta-item"><span class="meta-label">Tamaño</span><span class="meta-value">' +
                (artifact.size_bytes ? fmt(artifact.size_bytes) + ' B' : '—') + '</span></div>' +
              '<div class="meta-item"><span class="meta-label">Estado</span>' +
                (exists ? '<span class="status-badge status-succeeded"><span class="dot"></span>En disk</span>'
                        : '<span class="status-badge status-failed"><span class="dot"></span>Perdido</span>') + '</div>' +
              '<div class="meta-item artifact-path"><span class="meta-label">Archivo</span>' +
                '<code class="artifact-path-value" title="' + esc(artifact.path || '') + '">' +
                  esc(artifact.path || '') + '</code></div>' +
            '</div>' +
            (exists ? '' : '<p class="hint">El archivo del artefacto ya no existe (puede haberse borrado al eliminar el job).</p>') +
          '</div>' +
        '</section>';
    }

    var logPanel =
      '<section class="panel" aria-label="Registro del job">' +
        '<div class="panel-header"><span class="panel-icon" aria-hidden="true">≡</span>' +
        '<h2 class="panel-title">Registro</h2></div>' +
        '<div class="panel-body"><ul class="log-list">' +
          (job.log || []).slice().reverse().map(function (entry) {
            return '<li class="log-item log-' + esc(entry.level) + '">' +
              '<span class="log-time mono">' + esc(isoDate(entry.at)) + '</span>' +
              '<span class="log-msg">' + esc(entry.message) + '</span></li>';
          }).join('') +
        '</ul></div>' +
      '</section>';

    var backLink =
      '<a class="link-back" href="/training">← Todos los jobs</a>';

    dom.detail.innerHTML = backLink + header + metaGrid + errorBox + resultPanel + artifactPanel + logPanel;
    wireDetailActions(job);
    if (job.status === 'pending' || job.status === 'running') pollJob(job.job_id);
  }

  function pollJob(jobId) {
    stopPolling();
    var poll = function () {
      return api('/api/training/jobs/' + encodeURIComponent(jobId)).then(function (data) {
        state.job = data.job || data;
        renderJobDetail(state.job);
        if (state.job && (state.job.status === 'pending' || state.job.status === 'running')) {
          state.pollTimer = window.setTimeout(poll, 2000);
        } else {
          stopPolling();
        }
      });
    };
    poll();
  }

  function wireDetailActions(job) {
    if (!dom.detail) return;
    var cancel = dom.detail.querySelector('[data-action="detail-cancel"]');
    if (cancel) cancel.addEventListener('click', function () {
      confirmCallback = function () { return cancelJob(job.job_id); };
      openConfirm('Cancelar job', 'El job en cola se cancelará y no llegará a ejecutarse.');
    });
    var del = dom.detail.querySelector('[data-action="detail-delete"]');
    if (del) del.addEventListener('click', function () {
      confirmCallback = function () { return deleteJob(job.job_id); };
      openConfirm('Eliminar job', 'El job y (si es fit) su artefacto se eliminarán del servidor.');
    });
    var delArtifact = dom.detail.querySelector('[data-action="delete-artifact"]');
    if (delArtifact) delArtifact.addEventListener('click', function () {
      var aid = job.artifact && job.artifact.artifact_id;
      confirmCallback = function () { return deleteArtifact(aid); };
      openConfirm('Borrar artefacto', 'El archivo .pkl se borrará del disk. El job sigue en la lista.');
    });
  }

  function cancelJob(jobId) {
    return api('/api/training/jobs/' + encodeURIComponent(jobId) + '/cancel', {
      method: 'POST',
    }).then(function (data) {
      toast('Job cancelado');
      if (state.job && state.job.job_id === jobId) state.job.status = 'cancelled';
      renderJobDetail(state.job || {});
      return loadJobs();
    }).catch(function (error) { toast(error.message, true); });
  }

  function deleteJob(jobId) {
    return api('/api/training/jobs/' + encodeURIComponent(jobId), {
      method: 'DELETE',
    }).then(function () {
      toast('Job eliminado');
      state.job = null;
      window.location.href = '/training';
    }).catch(function (error) { toast(error.message, true); });
  }

  function deleteArtifact(artifactId) {
    if (!artifactId) return;
    return api('/api/training/artifacts/' + encodeURIComponent(artifactId), {
      method: 'DELETE',
    }).then(function () {
      toast('Artefacto borrado');
      renderJobDetail(state.job || {});
    }).catch(function (error) { toast(error.message, true); });
  }

  /* ------------------------------------------------------------------ */
  /* Create wizard                                                       */
  /* ------------------------------------------------------------------ */

  function loadModels() {
    if (!dom.modelSelect) return;
    return api('/api/training/models').then(function (data) {
      state.models = data.models || [];
      var options = ['<option value="">— selecciona un modelo —</option>'];
      state.models.forEach(function (model) {
        options.push('<option value="' + esc(model.id) + '">' + esc(model.name) +
          ' · ' + esc(model.id) + '</option>');
      });
      dom.modelSelect.innerHTML = options.join('');
    }).catch(function (error) {
      toast('No se pudieron cargar los modelos: ' + error.message, true);
    });
  }

  function modelById(id) {
    return state.models.find(function (model) { return model.id === id; });
  }

  function onKindChange() {
    state.wKind = dom.kindSelect ? dom.kindSelect.value : 'fit';
    state.wMethod = state.wKind === 'fit' ? '' : 'score';
    if (dom.kindHint) {
      dom.kindHint.textContent = state.wKind === 'evaluate'
        ? 'Evaluate evalúa un artefacto de fit previo (o el constructor si no se indica).'
        : 'Fit construye y entrena el modelo desde cero.';
    }
    // artifact field only applies to evaluate
    if (dom.artifactField) {
      dom.artifactField.hidden = state.wKind !== 'evaluate';
    }
    // reset method when switching kind
    if (dom.methodInput) dom.methodInput.value = state.wKind === 'evaluate' ? 'score' : '';
    updateWizardStatus();
    if (state.wModel) onModelChange();
  }

  function onModelChange() {
    var id = dom.modelSelect ? dom.modelSelect.value : '';
    state.wModel = id;
    var model = modelById(id);
    if (dom.methodInput) {
      dom.methodInput.value = model ? (state.wKind === 'evaluate' ? 'score' : model.fit_method) : '';
    }
    // constructor fields hint
    if (dom.constructorHint) {
      if (model && model.constructor_fields && model.constructor_fields.length) {
        dom.constructorHint.textContent = 'Constructor: ' + model.constructor_fields.map(esc).join(', ');
      } else {
        dom.constructorHint.textContent = 'El constructor de este modelo no requiere parámetros.';
      }
    }
    if (dom.methodHint) {
      if (model && model.method_fields && model.method_fields.length) {
        var required = model.method_fields.filter(function (f) { return f.required; });
        dom.methodHint.textContent = 'Método: ' + (model.fit_method || '') + ' · obligatorio: ' +
          required.map(function (f) { return esc(f.name); }).join(', ');
      } else {
        dom.methodHint.textContent = 'El método no requiere argumentos adicionales.';
      }
    }
    updateWizardStatus();
  }

  function openWizard(presetModelId) {
    loadModels().then(function () {
      if (dom.createDialog) {
        dom.createDialog.hidden = false;
        dom.createDialog.classList.add('is-open');
      }
      // reset
      state.wStep = 1;
      state.wKind = 'fit';
      state.wModel = presetModelId || '';
      state.wMethod = '';
      state.wName = '';
      state.wTimeout = '';
      state.wParams = '{}';
      state.wArgs = '{}';
      state.wArtifact = '';
      if (dom.kindSelect) dom.kindSelect.value = state.wKind;
      if (dom.modelSelect) dom.modelSelect.value = state.wModel;
      onKindChange();
      if (dom.nameInput) dom.nameInput.value = '';
      if (dom.timeoutInput) dom.timeoutInput.value = '';
      if (dom.parametersInput) dom.parametersInput.value = '{}';
      if (dom.argumentsInput) dom.argumentsInput.value = '{}';
      if (dom.artifactInput) dom.artifactInput.value = '';
      setStep(1);
    });
  }

  function closeWizard() {
    if (dom.createDialog) {
      dom.createDialog.classList.remove('is-open');
      dom.createDialog.hidden = true;
    }
  }

  function setStep(n) {
    state.wStep = n;
    if (dom.panel1) dom.panel1.hidden = n !== 1;
    if (dom.panel2) dom.panel2.hidden = n !== 2;
    each(document.querySelectorAll('#wizard-steps .wizard-step'), function (button) {
      var step = Number(button.getAttribute('data-step'));
      button.classList.toggle('is-active', step === n);
      button.classList.toggle('is-done', step < n);
      button.disabled = step > n;
      button.setAttribute('aria-selected', step === n ? 'true' : 'false');
    });
    if (dom.wizardBack) dom.wizardBack.disabled = n === 1;
    if (dom.wizardCreate) {
      dom.wizardCreate.textContent = n < 2 ? 'Siguiente paso' : 'Lanzar job';
      dom.wizardCreate.disabled = false;
    }
    updateWizardStatus();
  }

  function nextStep() {
    // basic validation for step 1
    if (state.wStep === 1) {
      if (!state.wModel) { updateWizardStatus('Selecciona un modelo.'); return; }
      var model = modelById(state.wModel);
      state.wMethod = dom.methodInput ? dom.methodInput.value.trim() : (model ? model.fit_method : '');
      if (!state.wMethod) { updateWizardStatus('Indica un método (fit/train por defecto).'); return; }
    }
    if (state.wStep < 2) setStep(state.wStep + 1);
  }

  function updateWizardStatus(message) {
    if (!dom.wizardStatus) return;
    dom.wizardStatus.textContent = message || '';
  }

  function createJob() {
    var body = {
      kind: state.wKind,
      model_id: state.wModel,
      method: state.wMethod || undefined,
      name: state.wName || undefined,
      timeout: state.wTimeout || undefined,
    };
    var params = {};
    try { params = JSON.parse(state.wParams || '{}'); } catch (error) { toast('Parámetros no es JSON válido', true); return; }
    body.parameters = params;
    var args = {};
    try { args = JSON.parse(state.wArgs || '{}'); } catch (error) { toast('Argumentos no es JSON válido', true); return; }
    body.arguments = args;
    if (state.wKind === 'evaluate') body.artifact_id = state.wArtifact || undefined;

    var btn = dom.wizardCreate;
    if (btn) btn.disabled = true;
    api('/api/training/jobs', { method: 'POST', body: body }).then(function (data) {
      var job = data.job || data;
      toast('Job creado: ' + (job.name || job.job_id));
      closeWizard();
      window.location.href = job.page_url || ('/training/' + job.job_id);
    }).catch(function (error) {
      if (btn) btn.disabled = false;
      toast(error.message, true);
      updateWizardStatus(error.message);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Confirmation dialog                                                 */
  /* ------------------------------------------------------------------ */

  function openConfirm(title, text) {
    if (!dom.confirmDialog) return;
    if (byId('confirm-title')) byId('confirm-title').textContent = title;
    if (byId('confirm-text')) byId('confirm-text').textContent = text;
    dom.confirmDialog.classList.add('is-open');
    dom.confirmDialog.hidden = false;
  }

  function closeConfirm() {
    if (dom.confirmDialog) {
      dom.confirmDialog.classList.remove('is-open');
      dom.confirmDialog.hidden = true;
    }
    confirmCallback = null;
  }

  function acceptConfirm() {
    if (!confirmCallback) { closeConfirm(); return; }
    var callback = confirmCallback;
    var accept = byId('confirm-accept');
    if (accept) accept.disabled = true;
    Promise.resolve(callback()).then(function () {
      if (accept) accept.disabled = false;
      closeConfirm();
    }).catch(function (error) {
      if (accept) accept.disabled = false;
      toast(error.message, true);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Wiring and startup                                                  */
  /* ------------------------------------------------------------------ */

  function bindWizard() {
    var create = byId('create-job');
    if (create) create.addEventListener('click', function () { openWizard(); });
    var side = byId('side-new-job');
    if (side) side.addEventListener('click', function () { openWizard(); });
    var close = byId('create-close');
    if (close) close.addEventListener('click', closeWizard);
    var back = byId('wizard-back');
    if (back) back.addEventListener('click', function () {
      if (state.wStep > 1) setStep(state.wStep - 1);
    });
    var createBtn = byId('wizard-create');
    if (createBtn) {
      createBtn.addEventListener('click', function () {
        if (state.wStep < 2) nextStep();
        else createJob();
      });
    }
    if (dom.kindSelect) dom.kindSelect.addEventListener('change', onKindChange);
    if (dom.modelSelect) dom.modelSelect.addEventListener('change', onModelChange);
    if (dom.createDialog) {
      dom.createDialog.addEventListener('click', function (event) {
        if (event.target === dom.createDialog) closeWizard();
      });
    }
  }

  function bindDialogs() {
    var cancel = byId('confirm-cancel');
    if (cancel) cancel.addEventListener('click', closeConfirm);
    var accept = byId('confirm-accept');
    if (accept) accept.addEventListener('click', acceptConfirm);
    if (dom.confirmDialog) {
      dom.confirmDialog.addEventListener('click', function (event) {
        if (event.target === dom.confirmDialog) closeConfirm();
      });
    }
    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape') return;
      if (dom.confirmDialog && !dom.confirmDialog.hidden) closeConfirm();
      else if (dom.createDialog && !dom.createDialog.hidden) closeWizard();
    });
  }

  function bindFilters() {
    if (dom.search) dom.search.addEventListener('input', renderJobs);
    if (dom.kindFilter) dom.kindFilter.addEventListener('change', renderJobs);
    if (dom.statusFilter) dom.statusFilter.addEventListener('change', renderJobs);
    var refresh = byId('refresh-training');
    if (refresh) refresh.addEventListener('click', function () {
      loadJobs();
      loadArtifactCount();
      toast('Jobs actualizados');
    });
  }

  function boot() {
    dom.toast = byId('toast');
    dom.serverStatus = byId('server-status');
    dom.serverStatusText = byId('server-status-text');
    dom.sideVersion = byId('side-version');
    dom.listView = byId('training-list-view');
    dom.detailView = byId('training-detail-view');
    dom.detail = byId('job-detail');
    dom.kpis = byId('training-kpis');
    dom.results = byId('job-results');
    dom.search = byId('job-search');
    dom.kindFilter = byId('job-kind');
    dom.statusFilter = byId('job-status');
    dom.resultBar = byId('job-result-bar');
    dom.resultCount = byId('job-result-count');
    dom.resultHint = byId('job-result-hint');
    dom.countJobs = byId('count-jobs');
    dom.countArtifacts = byId('count-artifacts');
    dom.crumbCurrent = byId('crumb-current');
    dom.createDialog = byId('create-dialog');
    dom.confirmDialog = byId('confirm-dialog');
    dom.panel1 = byId('wizard-panel-1');
    dom.panel2 = byId('wizard-panel-2');
    dom.wizardBack = byId('wizard-back');
    dom.wizardCreate = byId('wizard-create');
    dom.wizardStatus = byId('wizard-status');
    dom.kindSelect = byId('job-kind-select');
    dom.modelSelect = byId('job-model-select');
    dom.methodInput = byId('job-method-input');
    dom.nameInput = byId('job-name');
    dom.timeoutInput = byId('job-timeout');
    dom.parametersInput = byId('job-parameters');
    dom.argumentsInput = byId('job-arguments');
    dom.artifactInput = byId('job-artifact-id');
    dom.artifactField = byId('artifact-field');
    dom.constructorHint = byId('constructor-hint');
    dom.methodHint = byId('method-hint');
    dom.kindHint = byId('kind-hint');

    bindSidebar();
    bindTheme();
    bindWizard();
    bindDialogs();
    bindFilters();
    loadHealth();
    loadModels();
    route();
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
