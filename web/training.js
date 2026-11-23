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
    dataset: null,
    datasetSelection: null,
  };

  var dom = {};
  var confirmCallback = null;

  function byId(id) { return document.getElementById(id); }

  function esc(value) {
    var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' };
    return String(value === null || value === undefined ? '' : value)
      .replace(/[&<>"']/g, function (ch) { return map[ch]; });
  }

  function cleanLog(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/\x1b\[[0-9;]*m/g, '')
      .replace(/\r/g, '\n');
  }

  function compactLog(value) {
    var text = cleanLog(value);
    var lines = text.split('\n').map(function (line) {
      return line.trim();
    }).filter(function (line) {
      if (!line) return false;
      if (/^\d+\/\d+\b/.test(line)) return false;
      if (/\d+\/\d+.*step/.test(line)) return false;
      if (/^Epoch \d+\/\d+$/.test(line)) return false;
      return true;
    });
    return lines.join('\n').trim() || text.trim();
  }

  function each(list, callback) { Array.prototype.forEach.call(list || [], callback); }

  function handleResponse(response) {
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
  }

  function api(path, options) {
    var config = options || {};
    var headers = { Accept: 'application/json' };
    if (config.body !== undefined) headers['Content-Type'] = 'application/json';
    return fetch(path, {
      method: config.method || 'GET',
      headers: headers,
      body: config.body === undefined ? undefined : JSON.stringify(config.body),
      credentials: 'same-origin',
    }).then(handleResponse);
  }

  function apiForm(path, form) {
    return fetch(path, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      body: form,
      credentials: 'same-origin',
    }).then(handleResponse);
  }

  function selectedValues(select) {
    if (!select) return [];
    return Array.prototype.filter.call(select.options, function (option) {
      return option.selected;
    }).map(function (option) { return option.value; });
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
      if (window.LMCharts) LMCharts.redrawAll();
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

  function historyKeys(history) {
    return Object.keys(history || {}).filter(function (key) {
      var values = history[key];
      return Array.isArray(values) && values.some(function (value) {
        return value !== null && value !== undefined && isFinite(Number(value));
      });
    });
  }

  function latestValue(values) {
    if (!Array.isArray(values)) return null;
    for (var index = values.length - 1; index >= 0; index -= 1) {
      var value = values[index];
      if (value !== null && value !== undefined && isFinite(Number(value))) {
        return Number(value);
      }
    }
    return null;
  }

  /* Friendly names for the Keras history keys that show up in training runs.
   * Unknown keys fall back to "snake case → Sentence case". */
  var METRIC_LABELS = {
    loss: 'Pérdida',
    val_loss: 'Pérdida (val)',
    accuracy: 'Exactitud',
    val_accuracy: 'Exactitud (val)',
    acc: 'Exactitud',
    val_acc: 'Exactitud (val)',
    precision: 'Precisión',
    val_precision: 'Precisión (val)',
    recall: 'Sensibilidad',
    val_recall: 'Sensibilidad (val)',
    mae: 'MAE',
    val_mae: 'MAE (val)',
    mse: 'MSE',
    val_mse: 'MSE (val)',
    rmse: 'RMSE',
    val_rmse: 'RMSE (val)',
    mape: 'MAPE',
    val_mape: 'MAPE (val)',
    auc: 'AUC',
    val_auc: 'AUC (val)',
    f1: 'F1',
    val_f1: 'F1 (val)',
    kappa: 'Kappa de Cohen',
    val_kappa: 'Kappa de Cohen (val)',
    cohen_kappa: 'Kappa de Cohen',
    val_cohen_kappa: 'Kappa de Cohen (val)',
    cohen_kappa_score: 'Kappa de Cohen',
    val_cohen_kappa_score: 'Kappa de Cohen (val)'
  };

  function metricLabel(key) {
    var name = String(key);
    var known = METRIC_LABELS[name.toLowerCase()];
    if (known) return known;
    var label = name.replace(/_/g, ' ');
    return label.charAt(0).toUpperCase() + label.slice(1);
  }

  var METRIC_COLORS = ['lava', 'blue', 'green', 'amber', 'navy', 'muted', 'faint'];

  function metricColorMap(keys) {
    var map = {};
    keys.forEach(function (key, index) {
      map[key] = METRIC_COLORS[index % METRIC_COLORS.length];
    });
    return map;
  }

  function isLossMetric(key) {
    return String(key).toLowerCase().indexOf('loss') >= 0;
  }

  function metricDirection(key) {
    var name = String(key).toLowerCase();
    return /(loss|error|mae|mse|rmse|mape)/.test(name) ? 'min' : 'max';
  }

  function isRateMetric(key) {
    var name = String(key).toLowerCase().replace(/^val_/, '');
    return name.indexOf('accuracy') >= 0 || name.indexOf('precision') >= 0 ||
      name.indexOf('recall') >= 0 || name.indexOf('f1') >= 0 ||
      name.indexOf('auc') >= 0;
  }

  function isKappaMetric(key) {
    return String(key).toLowerCase().indexOf('kappa') >= 0;
  }

  function formatMetric(key, value) {
    if (value === null || value === undefined || !isFinite(Number(value))) return '—';
    if (isRateMetric(key)) return (Number(value) * 100).toFixed(2) + '%';
    /* Cohen's kappa lives in [-1, 1]; four decimals keep the small changes
     * between epochs readable in the tiles and the summary table. */
    if (isKappaMetric(key)) {
      return window.LMCharts ? window.LMCharts.fmtNumber(value, 4) : String(value);
    }
    return fmt(value);
  }

  function metricBest(values, direction) {
    var best = null;
    var epoch = null;
    (values || []).forEach(function (value, index) {
      if (value === null || value === undefined || !isFinite(Number(value))) return;
      var numeric = Number(value);
      if (best === null || (direction === 'min' ? numeric < best : numeric > best)) {
        best = numeric;
        epoch = index + 1;
      }
    });
    return { value: best, epoch: epoch };
  }

  function metricTile(label, value, sub, color) {
    return '<div class="training-metric">' +
      '<span class="training-metric-label">' +
        (color
          ? '<span class="training-metric-dot" style="background: var(--db-' + esc(color) + ')"></span>'
          : '') +
        esc(label) +
      '</span>' +
      '<span class="training-metric-value">' + esc(value) + '</span>' +
      (sub ? '<span class="training-metric-sub">' + esc(sub) + '</span>' : '') +
      '</div>';
  }

  function trainingMetricsPanel(job) {
    var history = job.result && job.result.history;
    var keys = historyKeys(history);
    if (!keys.length) return '';
    var lossKeys = keys.filter(isLossMetric);
    var metricKeys = keys.filter(function (key) { return !isLossMetric(key); });
    var lossKey = lossKeys.indexOf('loss') >= 0 ? 'loss' : (lossKeys.length ? lossKeys[0] : keys[0]);
    var lossSummary = metricBest(history[lossKey], 'min');
    var epochs = keys.reduce(function (maximum, key) {
      return Math.max(maximum, history[key].length);
    }, 0);
    var colorMap = metricColorMap(keys);
    var lossColor = colorMap[lossKey] || 'lava';
    var tiles = [
      metricTile('Épocas', fmt(epochs), 'total'),
      metricTile(metricLabel(lossKey), formatMetric(lossKey, latestValue(history[lossKey])),
        'último valor', lossColor),
      metricTile('Mejor ' + metricLabel(lossKey), formatMetric(lossKey, lossSummary.value),
        lossSummary.epoch ? 'época ' + lossSummary.epoch : '', lossColor),
    ];
    metricKeys.forEach(function (key) {
      tiles.push(metricTile(metricLabel(key), formatMetric(key, latestValue(history[key])),
        'último valor', colorMap[key]));
    });

    /* Captions sit above each plot, in the same visual language as the catalog
     * cards: a colour swatch, a short title and monospaced micro-stats. */
    var lossFigure = lossKeys.length
      ? '<figure class="training-chart">' +
          '<figcaption class="training-chart-head">' +
            '<span class="training-chart-title">' +
              '<span class="training-chart-swatch" style="background: var(--db-' +
                esc(lossColor) + ')"></span>' +
              esc(metricLabel(lossKey)) +
            '</span>' +
            '<span class="training-chart-meta">' +
              (lossSummary.value !== null
                ? '<span class="training-chart-stat">mín. <b>' +
                    esc(formatMetric(lossKey, lossSummary.value)) + '</b></span>' +
                  (lossSummary.epoch
                    ? '<span class="training-chart-sep">·</span>' +
                      '<span class="training-chart-stat">época <b>' +
                      esc(lossSummary.epoch) + '</b></span>'
                    : '')
                : '<span class="training-chart-stat"><b>' + esc(epochs) +
                  '</b> épocas</span>') +
            '</span>' +
          '</figcaption>' +
          '<div class="training-loss-chart" id="training-loss-chart"></div>' +
        '</figure>'
      : '';
    var metricSwatches = metricKeys.slice(0, 4).map(function (key) {
      return '<span class="training-chart-swatch" style="background: var(--db-' +
        esc(colorMap[key] || 'blue') + ')"></span>';
    }).join('');
    var metricFigure = metricKeys.length
      ? '<figure class="training-chart">' +
          '<figcaption class="training-chart-head">' +
            '<span class="training-chart-title">' +
              (metricKeys.length > 1
                ? '<span class="training-chart-swatches" aria-hidden="true">' +
                  metricSwatches + '</span>'
                : '') +
              esc(metricKeys.length === 1 ? metricLabel(metricKeys[0]) : 'Métricas') +
            '</span>' +
            '<span class="training-chart-meta">' +
              '<span class="training-chart-stat"><b>' + esc(metricKeys.length) + '</b> ' +
                (metricKeys.length === 1 ? 'serie' : 'series') + '</span>' +
              '<span class="training-chart-sep">·</span>' +
              '<span class="training-chart-stat">última época <b>' +
                esc(epochs) + '</b></span>' +
            '</span>' +
          '</figcaption>' +
          '<div class="training-metrics-chart" id="training-metrics-chart"></div>' +
        '</figure>'
      : '';
    var kappaNote = keys.some(isKappaMetric)
      ? '<p class="training-note"><b>Kappa de Cohen</b>: corrige el acuerdo por azar ' +
        '(1 = acuerdo perfecto, 0 = azar). Con clases desbalanceadas la exactitud puede ' +
        'ser alta y kappa seguir en 0 si el modelo solo acierta la clase mayoritaria; ' +
        'mientras predice una única clase el valor es 0 por definición.</p>'
      : '';
    var control = job.control || {};
    var actionLabel = {
      stop: 'detener al alcanzar el objetivo',
      reduce_lr: 'reducir learning rate',
      both: 'detener y reducir learning rate',
    }[control.action] || control.action;
    var controlHint = control.metric
      ? '<p class="training-note">Control: <code>' + esc(control.metric) + '</code>' +
        (control.value !== null && control.value !== undefined ? ' · objetivo ' + esc(control.value) : '') +
        (actionLabel ? ' · ' + esc(actionLabel) : '') + '</p>'
      : '';
    var tableRows = keys.map(function (key) {
      var summary = metricBest(history[key], metricDirection(key));
      return '<tr>' +
        '<td><span class="training-metric-dot" style="background: var(--db-' +
          esc(colorMap[key] || 'blue') + ')"></span>' + esc(metricLabel(key)) + '</td>' +
        '<td>' + esc(formatMetric(key, latestValue(history[key]))) + '</td>' +
        '<td>' + esc(formatMetric(key, summary.value)) + '</td>' +
        '<td>' + (summary.epoch ? esc(summary.epoch) : '—') + '</td>' +
        '</tr>';
    }).join('');
    return '<section class="panel training-metrics-panel" aria-label="Monitor de entrenamiento">' +
      '<div class="panel-header">' +
        '<h2 class="panel-title"><span class="panel-icon" aria-hidden="true">◔</span>' +
        'Monitor de entrenamiento</h2>' +
        '<span class="training-metrics-badge">' + esc(epochs) + ' épocas</span></div>' +
      '<div class="panel-body">' +
        '<div class="training-metric-grid">' + tiles.join('') + '</div>' +
        lossFigure +
        metricFigure +
        kappaNote +
        '<div class="training-metric-table-wrap">' +
          '<table class="training-metric-table">' +
            '<thead><tr><th>Métrica</th><th>Último</th><th>Mejor</th><th>Época</th></tr></thead>' +
            '<tbody>' + tableRows + '</tbody>' +
          '</table>' +
        '</div>' +
        controlHint +
      '</div>' +
    '</section>';
  }

  function renderTrainingChart(job) {
    var history = job.result && job.result.history;
    if (!dom.detail || !history || !window.LMCharts) return;
    var keys = historyKeys(history);
    if (!keys.length) return;
    var epochs = keys.reduce(function (maximum, key) {
      return Math.max(maximum, history[key].length);
    }, 0);
    var colorMap = metricColorMap(keys);

    function pointsFor(seriesKeys) {
      var points = [];
      for (var epoch = 0; epoch < epochs; epoch += 1) {
        var point = { t: epoch + 1, epoch: epoch + 1 };
        seriesKeys.forEach(function (key) {
          var value = history[key][epoch];
          point[key] = value === null || value === undefined || !isFinite(Number(value))
            ? null
            : Number(value);
        });
        points.push(point);
      }
      return points;
    }

    /* Keras reports rates as fractions in [0, 1].  Only convert when the series
     * actually lives in that range, so a metric already expressed as a
     * percentage keeps its own units. */
    function percentSeries(key) {
      var peak = 0;
      (history[key] || []).forEach(function (value) {
        if (value === null || value === undefined) return;
        var numeric = Number(value);
        if (isFinite(numeric)) peak = Math.max(peak, Math.abs(numeric));
      });
      return isRateMetric(key) && peak <= 1;
    }

    function seriesValueFormatter(key) {
      var percent = percentSeries(key);
      return function (value) {
        if (percent) return window.LMCharts.fmtNumber(Number(value) * 100, 2) + '%';
        return window.LMCharts.fmtNumber(value, 4);
      };
    }

    function seriesFor(seriesKeys) {
      return seriesKeys.map(function (key) {
        return {
          key: key,
          label: metricLabel(key),
          color: colorMap[key] || 'blue',
          /* A single curve gets the soft gradient wash; stacking several washes
           * would muddy the plot. */
          area: seriesKeys.length === 1,
          /* Tooltips and end labels follow the units of each series. */
          formatValue: seriesValueFormatter(key)
        };
      });
    }

    /* The plot follows the card width instead of a fixed pixel height, so the
     * same figure reads well on a phone and on a wide desktop. */
    function draw(containerId, seriesKeys, title, ariaLabel, options) {
      options = options || {};
      var container = dom.detail.querySelector('#' + containerId);
      if (!container || !seriesKeys.length) return;
      var allPercent = seriesKeys.every(percentSeries);
      window.LMCharts.line(container, {
        points: pointsFor(seriesKeys),
        series: seriesFor(seriesKeys),
        aspectRatio: options.aspectRatio || 0.5,
        minHeight: options.minHeight || 168,
        maxHeight: options.maxHeight || 300,
        title: title,
        smooth: true,
        legend: seriesKeys.length > 1,
        /* The final value is printed next to the head of a single curve. */
        endLabels: seriesKeys.length === 1,
        xAxisTitle: 'Época',
        tooltipTitle: function (point) { return 'Época ' + point.epoch; },
        formatXAxis: function (point) { return String(point.epoch); },
        formatAxis: allPercent
          ? function (value) { return window.LMCharts.fmtNumber(Number(value) * 100, 0) + '%'; }
          : function (value) { return window.LMCharts.fmtCompact(value); },
        formatValue: function (value) { return window.LMCharts.fmtNumber(value, 4); },
        emptyText: 'La curva se mostrará cuando el job avance.',
        ariaLabel: ariaLabel
      });
    }

    draw('training-loss-chart', keys.filter(isLossMetric), 'Pérdida por época',
      'Evolución de la pérdida por época', { aspectRatio: 0.48, maxHeight: 300 });
    draw('training-metrics-chart', keys.filter(function (key) { return !isLossMetric(key); }),
      'Métricas por época', 'Evolución de las métricas por época',
      { aspectRatio: 0.52, maxHeight: 300 });
  }

  function renderJobDetail(job) {
    if (!dom.detail) return;
    var isFit = job.kind === 'fit';
    var artifact = job.artifact || null;
    var model = job.model || null;
    var instance = job.instance || {};
    var paramsText = JSON.stringify(job.parameters || {}, null, 2);
    var argsText = JSON.stringify(job.arguments || {}, null, 2);
    var compileText = JSON.stringify(job.compile || {}, null, 2);

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
      (job.adapter ? '<div class="meta-item"><span class="meta-label">Adaptador</span><code>' + esc(job.adapter) + '</code></div>' : '') +
      (job.dataset ? '<div class="meta-item"><span class="meta-label">Dataset</span><span class="meta-value">' +
        esc(job.dataset.name) + '</span><span class="meta-sub mono">' + esc(job.dataset.summary || '') + '</span></div>' : '') +
      (job.normalization && job.normalization !== 'none'
        ? '<div class="meta-item"><span class="meta-label">Normalización</span><code>' +
          esc(job.normalization) + '</code></div>'
        : '') +
      (job.target_normalization && job.target_normalization.method &&
       job.target_normalization.method !== 'none'
        ? '<div class="meta-item"><span class="meta-label">Normalización Y</span><code>' +
          esc(job.target_normalization.method) + '</code></div>'
        : '') +
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
          '<div class="panel-header"><h2 class="panel-title">' +
          '<span class="panel-icon" aria-hidden="true">⚠</span>El job falló</h2></div>' +
          '<div class="panel-body"><pre class="error-pre">' + esc(cleanLog(job.error)) + '</pre>' +
          (job.error_detail
            ? '<details class="details-block"><summary>Detalle técnico</summary>' +
              '<pre class="code-block">' + esc(cleanLog(job.error_detail)) + '</pre></details>'
            : '') +
          '</div>' +
        '</section>';
    }

    var resultPanel = '';
    if (job.status === 'succeeded') {
      resultPanel =
        '<section class="panel" aria-label="Resultado">' +
          '<div class="panel-header"><h2 class="panel-title">' +
          '<span class="panel-icon" aria-hidden="true">✓</span>Resultado</h2></div>' +
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
            (job.result && job.result.history
              ? '<details class="details-block"><summary>Historial de entrenamiento (JSON)</summary>' +
                '<pre class="code-block">' + esc(JSON.stringify(job.result.history, null, 2)) + '</pre></details>'
              : '') +
            (Object.keys(job.compile || {}).length
              ? '<details class="details-block"><summary>Compilación Keras (JSON)</summary>' +
                '<pre class="code-block">' + esc(compileText) + '</pre></details>'
              : '') +
            (job.target_normalization && Object.keys(job.target_normalization).length
              ? '<details class="details-block"><summary>Escala de Y (JSON)</summary>' +
                '<pre class="code-block">' +
                esc(JSON.stringify(job.target_normalization, null, 2)) + '</pre></details>'
              : '') +
          '</div>' +
        '</section>';
    }

    var metricsPanel = job.status === 'succeeded' ? trainingMetricsPanel(job) : '';

    var artifactPanel = '';
    if (artifact) {
      var exists = artifact.exists === true;
      artifactPanel =
        '<section class="panel artifact-panel" aria-label="Artefacto">' +
          '<div class="panel-header">' +
            '<h2 class="panel-title"><span class="panel-icon" aria-hidden="true">{ }</span>Artefacto</h2>' +
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
              '<div class="meta-item artifact-path"><span class="meta-label">Carpeta</span>' +
                '<code class="artifact-path-value" title="' + esc(artifact.path || '') + '">' +
                  esc(artifact.path || '') + '</code></div>' +
              (artifact.load_path && artifact.load_path !== artifact.path
                ? '<div class="meta-item artifact-path"><span class="meta-label">Ruta de carga (Serving)</span>' +
                  '<code class="artifact-path-value" title="' + esc(artifact.load_path) + '">' +
                    esc(artifact.load_path) + '</code></div>'
                : '') +
            '</div>' +
            (artifact.files && artifact.files.length
              ? '<details class="details-block"><summary>Archivos del artefacto (' +
                esc(artifact.files.length) + ')</summary><pre class="code-block">' +
                esc(artifact.files.join('\n')) + '</pre></details>'
              : '') +
            (exists ? '' : '<p class="training-note">La carpeta del artefacto ya no existe (puede haberse borrado al eliminar el job).</p>') +
          '</div>' +
        '</section>';
    }

    var logPanel =
      '<section class="panel" aria-label="Registro del job">' +
        '<div class="panel-header"><h2 class="panel-title">' +
        '<span class="panel-icon" aria-hidden="true">≡</span>Registro</h2></div>' +
        '<div class="panel-body"><ul class="log-list">' +
          (job.log || []).slice().reverse().map(function (entry) {
            return '<li class="log-item log-' + esc(entry.level) + '">' +
              '<span class="log-time mono">' + esc(isoDate(entry.at)) + '</span>' +
              '<span class="log-msg">' + esc(compactLog(entry.message)) + '</span></li>';
          }).join('') +
        '</ul></div>' +
      '</section>';

    var backLink =
      '<a class="link-back" href="/training">← Todos los jobs</a>';

    dom.detail.innerHTML = backLink + header + metaGrid + errorBox + resultPanel +
      metricsPanel + artifactPanel + logPanel;
    wireDetailActions(job);
    renderTrainingChart(job);
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
      openConfirm('Borrar artefacto', 'La carpeta del artefacto se borrará del disk. El job sigue en la lista.');
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
        var suffix = model.example && model.example.label
          ? ' · ' + esc(model.example.label)
          : (model.category ? ' · ' + esc(model.category) : '');
        options.push('<option value="' + esc(model.id) + '">' + esc(model.name) + suffix + '</option>');
      });
      dom.modelSelect.innerHTML = options.join('');
    }).catch(function (error) {
      toast('No se pudieron cargar los modelos: ' + error.message, true);
    });
  }

  function modelById(id) {
    return state.models.find(function (model) { return model.id === id; });
  }

  function controlAction() {
    return dom.targetAction ? dom.targetAction.value : 'stop';
  }

  function syncControlFields() {
    var enabled = dom.controlEnabled && dom.controlEnabled.checked;
    if (dom.controlOptions) dom.controlOptions.hidden = !enabled;
    var action = controlAction();
    var stop = action === 'stop' || action === 'both';
    var reduce = action === 'reduce_lr' || action === 'both';
    if (dom.targetValueField) dom.targetValueField.hidden = !enabled || !stop;
    if (dom.targetFactorField) dom.targetFactorField.hidden = !enabled || !reduce;
    if (dom.targetMinLrField) dom.targetMinLrField.hidden = !enabled || !reduce;
  }

  function syncControlVisibility() {
    var model = modelById(state.wModel);
    if (dom.controlField) {
      dom.controlField.hidden = !(
        state.wKind === 'fit' &&
        model &&
        model.adapter === 'keras'
      );
    }
    syncControlFields();
  }

  function readTrainingControl() {
    if (!dom.controlField || dom.controlField.hidden) return null;
    if (!dom.controlEnabled || !dom.controlEnabled.checked) return null;
    var metric = dom.targetMetric ? dom.targetMetric.value : '';
    if (!metric) return null;
    var action = controlAction();
    var stop = action === 'stop' || action === 'both';
    var reduce = action === 'reduce_lr' || action === 'both';
    var control = {
      metric: metric,
      mode: dom.targetMode ? dom.targetMode.value : 'auto',
      action: action,
      patience: dom.targetPatience ? Number(dom.targetPatience.value || 3) : 3,
      min_delta: dom.targetMinDelta ? Number(dom.targetMinDelta.value || 0) : 0,
      max_epochs: dom.targetEpochs ? Number(dom.targetEpochs.value || 50) : 50,
    };
    if (stop) {
      var valueText = dom.targetValue ? String(dom.targetValue.value).trim() : '';
      if (!valueText) {
        throw new Error('Indica el valor objetivo de la métrica o cambia la acción a reducir LR.');
      }
      control.value = Number(valueText);
      if (!isFinite(control.value)) {
        throw new Error('El valor objetivo debe ser un número.');
      }
    }
    if (reduce) {
      control.factor = dom.targetFactor ? Number(dom.targetFactor.value || 0.5) : 0.5;
      control.min_lr = dom.targetMinLr ? Number(dom.targetMinLr.value || 0.000001) : 0.000001;
      if (!isFinite(control.factor) || control.factor <= 0 || control.factor >= 1) {
        throw new Error('El factor de reducción LR debe estar entre 0 y 1.');
      }
      if (!isFinite(control.min_lr) || control.min_lr < 0) {
        throw new Error('El learning rate mínimo debe ser un número no negativo.');
      }
    }
    return control;
  }

  function resetTrainingControl() {
    if (dom.controlEnabled) dom.controlEnabled.checked = false;
    if (dom.targetMetric) dom.targetMetric.value = 'loss';
    if (dom.targetMode) dom.targetMode.value = 'auto';
    if (dom.targetValue) dom.targetValue.value = '';
    if (dom.targetPatience) dom.targetPatience.value = '3';
    if (dom.targetMinDelta) dom.targetMinDelta.value = '0';
    if (dom.targetEpochs) dom.targetEpochs.value = '50';
    if (dom.targetAction) dom.targetAction.value = 'stop';
    if (dom.targetFactor) dom.targetFactor.value = '0.5';
    if (dom.targetMinLr) dom.targetMinLr.value = '0.000001';
    syncControlFields();
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
    if (dom.compileField && state.wKind !== 'fit') dom.compileField.hidden = true;
    if (dom.normalizationField) dom.normalizationField.hidden = state.wKind !== 'fit';
    if (dom.targetNormalizationField) dom.targetNormalizationField.hidden = state.wKind !== 'fit';
    syncControlVisibility();
    if (dom.datasetPanel && state.wKind !== 'fit') dom.datasetPanel.hidden = true;
    // reset method when switching kind
    if (dom.methodInput) dom.methodInput.value = state.wKind === 'evaluate' ? 'score' : '';
    updateWizardStatus();
    if (state.wModel) onModelChange();
  }

  function onModelChange() {
    var previous = state.wModel;
    var id = dom.modelSelect ? dom.modelSelect.value : '';
    state.wModel = id;
    if (previous !== id) resetDataset();
    var model = modelById(id);
    if (dom.methodInput) {
      dom.methodInput.value = model ? (state.wKind === 'evaluate' ? 'score' : model.fit_method) : '';
    }
    if (dom.compileField) {
      dom.compileField.hidden = !(state.wKind === 'fit' && model && model.adapter === 'keras');
    }
    if (dom.compileInput) {
      var compileTask = model && model.dataset_task;
      dom.compileInput.placeholder = compileTask === 'regression'
        ? '{"optimizer": "adam", "loss": "mse", "metrics": ["mae", "mse", "root_mean_squared_error", "r2_score"]}'
        : (compileTask === 'classification'
          ? '{"optimizer": "adam", "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy", "precision", "recall", "f1_score", "auc", "kappa"]}'
          : '{"optimizer": "adam", "loss": "mse", "metrics": ["mae"]}');
    }
    if (dom.targetNormalizationField) {
      dom.targetNormalizationField.hidden =
        !(state.wKind === 'fit' && model && model.dataset_task !== 'classification');
    }
    syncControlVisibility();
    var datasetMode = model && model.dataset_mode;
    var datasetVisible = state.wKind === 'fit' && datasetMode && datasetMode !== 'generic';
    if (dom.datasetPanel) dom.datasetPanel.hidden = !datasetVisible;
    if (dom.datasetModeBadge) {
      dom.datasetModeBadge.textContent = datasetMode === 'xy'
        ? 'X + Y'
        : (datasetMode === 'datapoints' ? 'datapoints' : '');
    }
    if (dom.datasetXField) dom.datasetXField.hidden = datasetMode === 'datapoints';
    if (dom.datasetYLabel) {
      dom.datasetYLabel.textContent = datasetMode === 'datapoints'
        ? 'Columna de datapoints'
        : 'Columna Y / objetivo';
    }
    if (state.dataset) {
      renderDatasetColumns(state.dataset);
    } else {
      syncDatasetTarget();
    }
    if (dom.modelCard) {
      dom.modelCard.hidden = !model;
      if (model) {
        if (dom.modelIcon) dom.modelIcon.textContent = model.icon || '🧪';
        if (dom.modelName) dom.modelName.textContent = model.name || model.id;
        if (dom.modelMeta) {
          dom.modelMeta.textContent = [
            model.example && model.example.label ? model.example.label : '',
            model.fit_method ? 'método ' + model.fit_method : '',
            model.adapter ? 'adapter ' + model.adapter : '',
            model.category || '',
          ].filter(Boolean).join(' · ');
        }
        if (dom.modelNote) {
          dom.modelNote.textContent = model.example && model.example.note
            ? model.example.note
            : 'Selecciona el ejemplo o escribe los datos manualmente.';
        }
      }
    }
    if (dom.wizardExample) {
      dom.wizardExample.hidden = !(state.wKind === 'fit' && model && model.example);
      dom.wizardExample.textContent = 'Cargar ejemplo';
    }
    if (dom.exampleNote) {
      var note = state.wKind === 'fit' && model && model.example ? model.example.note : '';
      dom.exampleNote.hidden = !note;
      dom.exampleNote.textContent = note || '';
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
        var adapter = model.adapter ? ' · adapter: ' + esc(model.adapter) : '';
        dom.methodHint.textContent = 'Método: ' + (model.fit_method || '') + ' · obligatorio: ' +
          required.map(function (f) { return esc(f.name); }).join(', ') + adapter;
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
      if (dom.compileInput) dom.compileInput.value = '';
      if (dom.optimizerSelect) dom.optimizerSelect.value = 'auto';
      if (dom.learningRateInput) dom.learningRateInput.value = '';
      if (dom.compileField) dom.compileField.hidden = true;
      if (dom.normalizationSelect) dom.normalizationSelect.value = 'none';
      if (dom.targetNormalizationSelect) dom.targetNormalizationSelect.value = 'auto';
      resetTrainingControl();
      if (dom.artifactInput) dom.artifactInput.value = '';
      if (dom.modelCard) dom.modelCard.hidden = true;
      if (dom.wizardExample) { dom.wizardExample.hidden = true; dom.wizardExample.textContent = 'Cargar ejemplo'; }
      if (dom.exampleNote) { dom.exampleNote.hidden = true; dom.exampleNote.textContent = ''; }
      resetDataset();
      if (state.wModel) onModelChange();
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

  function setDatasetStatus(message, isError) {
    if (!dom.datasetStatus) return;
    dom.datasetStatus.textContent = message || '';
    dom.datasetStatus.classList.toggle('is-error', Boolean(isError));
  }

  function resetDataset() {
    state.dataset = null;
    state.datasetSelection = null;
    if (dom.datasetPanel) dom.datasetPanel.hidden = true;
    if (dom.datasetConfig) dom.datasetConfig.hidden = true;
    if (dom.datasetSummary) dom.datasetSummary.textContent = '';
    if (dom.datasetPreview) dom.datasetPreview.innerHTML = '';
    if (dom.datasetX) dom.datasetX.innerHTML = '';
    if (dom.datasetY) dom.datasetY.innerHTML = '';
    if (dom.datasetApplyNote) dom.datasetApplyNote.textContent = '';
    if (dom.datasetFile) dom.datasetFile.value = '';
    setDatasetStatus('');
  }

  function renderDatasetPreview(dataset) {
    if (!dom.datasetPreview || !dataset || !dataset.columns) return;
    var columns = dataset.columns.map(function (column) { return column.name; });
    var head = '<thead><tr>' + columns.map(function (name) {
      return '<th>' + esc(name) + '</th>';
    }).join('') + '</tr></thead>';
    var body = '<tbody>' + (dataset.preview || []).map(function (row) {
      return '<tr>' + columns.map(function (name) {
        var value = row[name];
        return '<td>' + esc(value === null || value === undefined ? '—' : value) + '</td>';
      }).join('') + '</tr>';
    }).join('') + '</tbody>';
    dom.datasetPreview.innerHTML = head + body;
  }

  function preferredTarget(dataset, model) {
    var columns = dataset.columns || [];
    var numeric = columns.filter(function (column) { return column.numeric; });
    var mode = model && model.dataset_mode;
    var task = model && model.dataset_task;
    var preferred;
    if (mode === 'datapoints') {
      preferred = ['close', 'value', 'price', 'target', 'y'];
    } else if (task === 'classification') {
      preferred = ['species', 'target', 'class', 'label', 'y'];
    } else {
      preferred = ['median_house_value', 'target', 'price', 'value', 'y'];
    }
    var candidates = (mode === 'datapoints' || task === 'regression') ? numeric : columns;
    for (var index = 0; index < preferred.length; index += 1) {
      var found = candidates.find(function (column) { return column.name === preferred[index]; });
      if (found) return found;
    }
    return candidates.length ? candidates[candidates.length - 1] : null;
  }

  function renderDatasetColumns(dataset) {
    var model = modelById(state.wModel);
    if (!model || !dataset || !dataset.columns) return;
    var mode = model.dataset_mode;
    var numeric = dataset.columns.filter(function (column) { return column.numeric; });
    var target = preferredTarget(dataset, model);
    var targetCandidates = (mode === 'datapoints' || model.dataset_task === 'regression')
      ? numeric
      : dataset.columns;
    if (dom.datasetY) {
      dom.datasetY.innerHTML = targetCandidates.map(function (column) {
        return '<option value="' + esc(column.name) + '">' + esc(column.name) +
          ' · ' + esc(column.dtype) + '</option>';
      }).join('');
      if (target) dom.datasetY.value = target.name;
    }
    if (dom.datasetXField) dom.datasetXField.hidden = mode === 'datapoints';
    if (mode !== 'datapoints' && dom.datasetX) {
      var targetName = target ? target.name : '';
      dom.datasetX.innerHTML = numeric.map(function (column) {
        var selected = column.name !== targetName;
        return '<option value="' + esc(column.name) + '"' + (selected ? ' selected' : '') +
          '>' + esc(column.name) + ' · ' + esc(column.dtype) + '</option>';
      }).join('');
    }
    syncDatasetTarget();
  }

  function syncDatasetTarget() {
    var model = modelById(state.wModel);
    var mode = model && model.dataset_mode;
    var target = dom.datasetY ? dom.datasetY.value : '';
    if (mode === 'xy' && dom.datasetX) {
      Array.prototype.forEach.call(dom.datasetX.options, function (option) {
        if (option.value === target) option.selected = false;
      });
    }
    if (dom.datasetTargetHint) {
      if (mode === 'datapoints') {
        dom.datasetTargetHint.textContent = 'La columna se usará como secuencia de datapoints.';
      } else if (model && model.dataset_task === 'classification') {
        dom.datasetTargetHint.textContent = 'Las etiquetas de texto se codifican como enteros (0..n-1).';
      } else if (model && model.dataset_task === 'auto') {
        dom.datasetTargetHint.textContent = 'Se detecta automáticamente: texto → clasificación, numérico → regresión.';
      } else {
        dom.datasetTargetHint.textContent = 'La columna objetivo debe ser numérica.';
      }
    }
    if (dom.datasetApplyNote) dom.datasetApplyNote.textContent = '';
  }

  function setDataset(dataset) {
    if (!dataset) {
      setDatasetStatus('La respuesta no contiene un dataset válido.', true);
      return;
    }
    state.dataset = dataset;
    state.datasetSelection = null;
    if (dom.datasetConfig) dom.datasetConfig.hidden = false;
    if (dom.datasetSummary) {
      dom.datasetSummary.textContent = dataset.name + ' · ' + dataset.rows + ' filas · ' +
        dataset.columns.length + ' columnas · ' + String(dataset.format || '').toUpperCase();
    }
    renderDatasetPreview(dataset);
    renderDatasetColumns(dataset);
    setDatasetStatus('Dataset cargado. Selecciona las columnas y pulsa «Aplicar selección».');
  }

  function loadDatasetUrl(url) {
    var text = String(url || '').trim();
    if (!text) {
      toast('Escribe una URL de dataset.', true);
      return;
    }
    setDatasetStatus('Descargando y analizando el dataset…');
    api('/api/training/datasets/inspect', { method: 'POST', body: { url: text } }).then(function (data) {
      setDataset(data.dataset);
      toast('Dataset cargado: ' + (data.dataset ? data.dataset.name : text));
    }).catch(function (error) {
      setDatasetStatus(error.message, true);
      toast(error.message, true);
    });
  }

  function loadDatasetFile() {
    var file = dom.datasetFile && dom.datasetFile.files ? dom.datasetFile.files[0] : null;
    if (!file) {
      toast('Selecciona un archivo CSV, XLSX o Parquet.', true);
      return;
    }
    var form = new FormData();
    form.append('file', file);
    setDatasetStatus('Subiendo y analizando ' + file.name + '…');
    apiForm('/api/training/datasets/upload', form).then(function (data) {
      setDataset(data.dataset);
      toast('Dataset cargado: ' + (data.dataset ? data.dataset.name : file.name));
    }).catch(function (error) {
      setDatasetStatus(error.message, true);
      toast(error.message, true);
    });
  }

  function datasetSelection() {
    var model = modelById(state.wModel);
    if (!state.dataset || !model) return null;
    var mode = model.dataset_mode || 'xy';
    return {
      dataset_id: state.dataset.dataset_id,
      mode: mode,
      task: model.dataset_task || 'auto',
      features: mode === 'datapoints' ? [] : selectedValues(dom.datasetX),
      target: dom.datasetY ? dom.datasetY.value : '',
    };
  }

  function validateDatasetSelection() {
    var selection = datasetSelection();
    if (!selection) {
      toast('Carga un dataset antes de lanzar el job.', true);
      return null;
    }
    if (!selection.target) {
      toast('Selecciona la columna Y u objetivo.', true);
      return null;
    }
    if (selection.mode === 'xy' && !selection.features.length) {
      toast('Selecciona al menos una columna X.', true);
      return null;
    }
    if (selection.mode === 'xy' && selection.features.indexOf(selection.target) >= 0) {
      toast('La columna objetivo no puede formar parte de X.', true);
      return null;
    }
    return selection;
  }

  function applyDataset() {
    var selection = validateDatasetSelection();
    if (!selection) return;
    state.datasetSelection = selection;
    if (dom.datasetApplyNote) {
      dom.datasetApplyNote.textContent = (selection.mode === 'datapoints'
        ? 'Selección lista: datapoints ← ' + selection.target
        : 'Selección lista: ' + selection.features.length + ' columnas X → ' + selection.target) +
        '. Se usará al lanzar el job.';
    }
    setDatasetStatus('Selección lista. Revisa los argumentos y lanza el job.');
    toast('Selección de dataset aplicada.');
  }

  function loadExample() {
    var model = modelById(state.wModel);
    if (!model || !model.example) {
      toast('Este modelo no tiene un ejemplo cargado.', true);
      return;
    }
    var example = model.example;
    if (example.method) state.wMethod = example.method;
    if (dom.methodInput && example.method) dom.methodInput.value = example.method;
    if (dom.nameInput && example.name) dom.nameInput.value = example.name;
    if (dom.parametersInput) {
      dom.parametersInput.value = JSON.stringify(example.parameters || {}, null, 2);
    }
    if (dom.argumentsInput) {
      dom.argumentsInput.value = JSON.stringify(example.arguments || {}, null, 2);
    }
    if (dom.compileInput) {
      dom.compileInput.value = example.compile && Object.keys(example.compile).length
        ? JSON.stringify(example.compile, null, 2)
        : '';
    }
    var exampleCompile = example.compile || {};
    var optimizerNames = ['adam', 'adamw', 'sgd', 'rmsprop', 'nadam', 'adagrad',
      'adadelta', 'adamax', 'ftrl'];
    var exampleOptimizer = String(exampleCompile.optimizer || '').toLowerCase();
    if (dom.optimizerSelect) {
      dom.optimizerSelect.value = optimizerNames.indexOf(exampleOptimizer) >= 0
        ? exampleOptimizer
        : 'auto';
    }
    if (dom.learningRateInput) {
      dom.learningRateInput.value =
        exampleCompile.learning_rate !== undefined && exampleCompile.learning_rate !== null
          ? exampleCompile.learning_rate
          : '';
    }
    if (dom.compileField) {
      dom.compileField.hidden = !(state.wKind === 'fit' && model.adapter === 'keras');
    }
    if (dom.exampleNote) {
      dom.exampleNote.hidden = !example.note;
      dom.exampleNote.textContent = example.note || '';
    }
    if (dom.wizardExample) dom.wizardExample.textContent = 'Ejemplo cargado';
    toast('Ejemplo cargado: ' + (example.label || model.name));
  }

  function createJob() {
    var nameText = dom.nameInput ? dom.nameInput.value.trim() : state.wName;
    var timeoutText = dom.timeoutInput ? dom.timeoutInput.value : state.wTimeout;
    var body = {
      kind: state.wKind,
      model_id: state.wModel,
      method: state.wMethod || undefined,
      name: nameText || undefined,
      timeout: timeoutText || undefined,
    };
    var paramsText = dom.parametersInput ? dom.parametersInput.value : state.wParams;
    var argsText = dom.argumentsInput ? dom.argumentsInput.value : state.wArgs;
    var params = {};
    try { params = JSON.parse(paramsText || '{}'); } catch (error) { toast('Parámetros no es JSON válido', true); return; }
    body.parameters = params;
    var args = {};
    try { args = JSON.parse(argsText || '{}'); } catch (error) { toast('Argumentos no es JSON válido', true); return; }
    body.arguments = args;
    if (dom.compileField && !dom.compileField.hidden) {
      var compile = {};
      var compileText = dom.compileInput ? dom.compileInput.value.trim() : '';
      if (compileText) {
        try { compile = JSON.parse(compileText); }
        catch (error) { toast('Compilación no es JSON válido', true); return; }
        if (!compile || typeof compile !== 'object' || Array.isArray(compile)) {
          toast('La compilación debe ser un objeto JSON', true);
          return;
        }
      }
      var optimizerValue = dom.optimizerSelect ? dom.optimizerSelect.value : 'auto';
      if (optimizerValue && optimizerValue !== 'auto' && optimizerValue !== 'custom') {
        compile.optimizer = optimizerValue;
      }
      var learningRateText = dom.learningRateInput ? dom.learningRateInput.value.trim() : '';
      if (learningRateText !== '') {
        var learningRate = Number(learningRateText);
        if (!isFinite(learningRate) || learningRate <= 0) {
          toast('El learning rate debe ser mayor que 0', true);
          return;
        }
        compile.learning_rate = learningRate;
      }
      if (Object.keys(compile).length) body.compile = compile;
    }
    try {
      var control = readTrainingControl();
      if (control) body.control = control;
    } catch (error) {
      toast(error.message, true);
      return;
    }
    if (state.wKind === 'fit' && dom.normalizationSelect) {
      body.normalization = dom.normalizationSelect.value;
    }
    if (state.wKind === 'fit' && dom.targetNormalizationSelect) {
      body.target_normalization = dom.targetNormalizationSelect.value;
    }
    if (state.dataset && dom.datasetPanel && !dom.datasetPanel.hidden) {
      var selection = validateDatasetSelection();
      if (!selection) return;
      body.dataset = selection;
    }
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

  function bindDataset() {
    if (dom.datasetLoadUrl) dom.datasetLoadUrl.addEventListener('click', function () {
      loadDatasetUrl(dom.datasetUrl ? dom.datasetUrl.value : '');
    });
    if (dom.datasetUrl) dom.datasetUrl.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') {
        event.preventDefault();
        loadDatasetUrl(dom.datasetUrl.value);
      }
    });
    if (dom.datasetLoadFile) dom.datasetLoadFile.addEventListener('click', loadDatasetFile);
    if (dom.datasetFile) dom.datasetFile.addEventListener('change', loadDatasetFile);
    if (dom.datasetExamples) dom.datasetExamples.addEventListener('click', function (event) {
      var chip = event.target.closest('[data-dataset-url]');
      if (!chip) return;
      var model = modelById(state.wModel);
      var chipTask = chip.getAttribute('data-dataset-task');
      if (model && chipTask && model.dataset_task && model.dataset_task !== 'auto' &&
          model.dataset_task !== chipTask) {
        toast('Este ejemplo es de ' + chipTask + '; el modelo espera ' + model.dataset_task + '.', true);
        return;
      }
      if (dom.datasetUrl) dom.datasetUrl.value = chip.getAttribute('data-dataset-url') || '';
      loadDatasetUrl(chip.getAttribute('data-dataset-url') || '');
    });
    if (dom.datasetY) dom.datasetY.addEventListener('change', syncDatasetTarget);
    if (dom.datasetX) dom.datasetX.addEventListener('change', function () {
      if (dom.datasetApplyNote) dom.datasetApplyNote.textContent = '';
    });
    if (dom.datasetApply) dom.datasetApply.addEventListener('click', applyDataset);
  }

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
    if (dom.controlEnabled) dom.controlEnabled.addEventListener('change', syncControlFields);
    if (dom.targetAction) dom.targetAction.addEventListener('change', syncControlFields);
    if (dom.wizardExample) dom.wizardExample.addEventListener('click', loadExample);
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
    dom.compileInput = byId('job-compile');
    dom.compileField = byId('compile-field');
    dom.optimizerSelect = byId('job-optimizer');
    dom.learningRateInput = byId('job-learning-rate');
    dom.normalizationSelect = byId('job-normalization');
    dom.normalizationField = byId('normalization-field');
    dom.targetNormalizationSelect = byId('job-target-normalization');
    dom.targetNormalizationField = byId('target-normalization-field');
    dom.controlField = byId('control-field');
    dom.controlEnabled = byId('job-control-enabled');
    dom.controlOptions = byId('control-options');
    dom.targetMetric = byId('job-target-metric');
    dom.targetMode = byId('job-target-mode');
    dom.targetValue = byId('job-target-value');
    dom.targetValueField = byId('target-value-field');
    dom.targetPatience = byId('job-target-patience');
    dom.targetMinDelta = byId('job-target-min-delta');
    dom.targetEpochs = byId('job-target-epochs');
    dom.targetAction = byId('job-target-action');
    dom.targetFactor = byId('job-target-factor');
    dom.targetFactorField = byId('target-factor-field');
    dom.targetMinLr = byId('job-target-min-lr');
    dom.targetMinLrField = byId('target-min-lr-field');
    dom.artifactInput = byId('job-artifact-id');
    dom.artifactField = byId('artifact-field');
    dom.constructorHint = byId('constructor-hint');
    dom.methodHint = byId('method-hint');
    dom.kindHint = byId('kind-hint');
    dom.modelCard = byId('wizard-model-card');
    dom.modelIcon = byId('wizard-model-icon');
    dom.modelName = byId('wizard-model-name');
    dom.modelMeta = byId('wizard-model-meta');
    dom.modelNote = byId('wizard-model-note');
    dom.wizardExample = byId('wizard-example');
    dom.exampleNote = byId('wizard-example-note');
    dom.datasetPanel = byId('dataset-panel');
    dom.datasetModeBadge = byId('dataset-mode-badge');
    dom.datasetUrl = byId('dataset-url');
    dom.datasetLoadUrl = byId('dataset-load-url');
    dom.datasetFile = byId('dataset-file');
    dom.datasetLoadFile = byId('dataset-load-file');
    dom.datasetExamples = document.querySelector('.dataset-examples');
    dom.datasetStatus = byId('dataset-status');
    dom.datasetConfig = byId('dataset-config');
    dom.datasetSummary = byId('dataset-summary');
    dom.datasetPreview = byId('dataset-preview');
    dom.datasetX = byId('dataset-x');
    dom.datasetXField = byId('dataset-x-field');
    dom.datasetY = byId('dataset-y');
    dom.datasetYLabel = byId('dataset-y-label');
    dom.datasetTargetHint = byId('dataset-target-hint');
    dom.datasetApply = byId('dataset-apply');
    dom.datasetApplyNote = byId('dataset-apply-note');

    bindSidebar();
    bindTheme();
    bindDataset();
    bindWizard();
    bindDialogs();
    bindFilters();
    loadHealth();
    loadModels();
    route();
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
