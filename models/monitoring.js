'use strict';

/*
 * Model Monitoring dashboard.
 *
 * Every value comes from /api/monitoring/*, which aggregates the serving event
 * log.  The page never invents traffic: an empty log renders zero KPIs and the
 * explicit empty states of each chart.
 */
(function () {
  var state = {
    range: '24h',
    customStart: '',
    customEnd: '',
    model: '',
    endpoint: '',
    eventStatus: '',
    data: null,
    detail: null,
    events: [],
    endpointRoute: '',
    sidebarEndpoints: [],
    auto: true,
    timer: 0,
    loading: false,
    lastUpdated: null
  };

  var dom = {};

  function byId(id) { return document.getElementById(id); }

  function esc(value) {
    var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' };
    return String(value === null || value === undefined ? '' : value)
      .replace(/[&<>"']/g, function (ch) { return map[ch]; });
  }

  function each(list, callback) {
    Array.prototype.forEach.call(list || [], callback);
  }

  function api(path) {
    return fetch(path, { headers: { Accept: 'application/json' }, credentials: 'same-origin' })
      .then(function (response) {
        return response.text().then(function (text) {
          var data = {};
          try { data = text ? JSON.parse(text) : {}; } catch (error) { data = { error: text }; }
          if (!response.ok || data.success === false) {
            var failure = new Error(data.error || ('HTTP ' + response.status));
            failure.status = response.status;
            failure.data = data;
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
    toast.timer = window.setTimeout(function () { dom.toast.classList.remove('is-visible'); }, 2800);
  }
  toast.timer = 0;

  function fmt(value) { return window.LMCharts ? LMCharts.fmtNumber(value) : String(value); }
  function compact(value) { return window.LMCharts ? LMCharts.fmtCompact(value) : String(value); }
  function ms(value) { return window.LMCharts ? LMCharts.fmtMs(value) : (value + ' ms'); }
  function percent(value, digits) { return window.LMCharts ? LMCharts.fmtPercent(value, digits) : (value + '%'); }

  function isoDate(value) {
    if (!value) return '—';
    var date = new Date(value);
    return isNaN(date.getTime()) ? String(value) : date.toLocaleString('es-ES');
  }

  function shortDate(value) {
    if (!value) return '—';
    var date = new Date(value);
    return isNaN(date.getTime()) ? String(value) : date.toLocaleString('es-ES', {
      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
    });
  }

  function emptyState(icon, title, text, action) {
    return '<div class="empty-state"><span class="empty-icon" aria-hidden="true">' + icon + '</span>' +
      '<h3>' + esc(title) + '</h3><p>' + esc(text) + '</p>' + (action || '') + '</div>';
  }

  function queryString(extra) {
    var params = new URLSearchParams();
    params.set('range', state.range);
    if (state.range === 'custom') {
      if (state.customStart) params.set('start', state.customStart);
      if (state.customEnd) params.set('end', state.customEnd);
    }
    if (state.model) params.set('model_id', state.model);
    if (state.endpoint) params.set('endpoint_id', state.endpoint);
    if (extra) Object.keys(extra).forEach(function (key) { params.set(key, extra[key]); });
    return params.toString();
  }

  /* ------------------------------------------------------------------ */
  /* Shell                                                               */
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

  function closeMobileSidebar() {
    document.body.classList.remove('sidebar-open');
    syncSidebarToggle();
  }

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
      if (window.LMCharts) LMCharts.redrawAll();
      toast(next === 'dark' ? 'Tema oscuro activado' : 'Tema claro activado');
    });
  }

  function loadHealth() {
    return api('/api/health').then(function (data) {
      if (dom.serverStatus) {
        dom.serverStatus.classList.remove('is-ok', 'is-warn', 'is-error');
        dom.serverStatus.classList.add('is-ok');
        var dot = dom.serverStatus.querySelector('.dot');
        if (dot) dot.className = 'dot dot--ok';
      }
      if (dom.serverStatusText) dom.serverStatusText.textContent = 'En línea';
      if (dom.sideVersion) dom.sideVersion.textContent = 'v' + (data.version || '—');
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

  /* ------------------------------------------------------------------ */
  /* Data loading                                                        */
  /* ------------------------------------------------------------------ */

  function loadMonitoring(initial) {
    if (state.loading) return;
    state.loading = true;
    if (initial && dom.loading) dom.loading.hidden = false;
    if (dom.error) dom.error.hidden = true;
    var refresh = byId('refresh-monitoring');
    if (refresh) refresh.classList.add('is-busy');

    var request = state.endpointRoute
      ? api('/api/monitoring/endpoints/' + encodeURIComponent(state.endpointRoute) + '?' + queryString())
      : Promise.all([
          api('/api/monitoring/overview?' + queryString()),
          api('/api/monitoring/events?' + queryString({
            limit: 50,
            status: state.eventStatus || ''
          }))
        ]);

    request.then(function (result) {
      state.loading = false;
      if (dom.loading) dom.loading.hidden = true;
      if (refresh) refresh.classList.remove('is-busy');
      if (state.endpointRoute) {
        state.detail = result;
        if (dom.globalView) dom.globalView.hidden = true;
        if (dom.endpointView) dom.endpointView.hidden = false;
        renderEndpointView(result);
      } else {
        state.data = result[0];
        state.events = result[1].events || [];
        if (dom.endpointView) dom.endpointView.hidden = true;
        if (dom.globalView) dom.globalView.hidden = false;
        renderGlobalView(state.data, result[1]);
        populateOptions();
        state.sidebarEndpoints = (state.data.options && state.data.options.endpoints) || [];
        updateSidebarEndpoints();
      }
      state.lastUpdated = new Date();
      if (dom.updated) dom.updated.textContent = 'Actualizado ' + state.lastUpdated.toLocaleTimeString('es-ES');
    }).catch(function (error) {
      state.loading = false;
      if (dom.loading) dom.loading.hidden = true;
      if (refresh) refresh.classList.remove('is-busy');
      if (dom.error) {
        dom.error.hidden = false;
        dom.error.innerHTML = emptyState(
          '⚠', 'No se pudieron cargar las métricas', error.message,
          '<button class="btn btn-secondary btn-sm" type="button" id="retry-monitoring">Reintentar</button>'
        );
        var retry = byId('retry-monitoring');
        if (retry) retry.addEventListener('click', function () { loadMonitoring(true); });
      }
    });
  }

  function populateOptions() {
    var data = state.data || {};
    var options = data.options || {};
    if (dom.modelFilter && options.models) {
      var currentModel = dom.modelFilter.value;
      dom.modelFilter.innerHTML = '<option value="">Todos los modelos</option>' +
        options.models.map(function (model) {
          return '<option value="' + esc(model.id) + '">' + esc(model.name) + '</option>';
        }).join('');
      dom.modelFilter.value = currentModel;
    }
    if (dom.endpointFilter && options.endpoints) {
      var currentEndpoint = dom.endpointFilter.value;
      dom.endpointFilter.innerHTML = '<option value="">Todos los endpoints</option>' +
        options.endpoints.map(function (endpoint) {
          return '<option value="' + esc(endpoint.id) + '">' + esc(endpoint.name) +
            (endpoint.exists ? '' : ' (histórico)') + '</option>';
        }).join('');
      dom.endpointFilter.value = currentEndpoint;
    }
  }

  function updateSidebarEndpoints() {
    var box = byId('side-endpoints');
    if (!box) return;
    var options = (state.data && state.data.options && state.data.options.endpoints) ||
      state.sidebarEndpoints || [];
    if (!options.length) {
      box.innerHTML = '<p class="empty" style="padding:4px 10px;font-size:10.5px">Sin endpoints</p>';
      return;
    }
    box.innerHTML = options.slice(0, 8).map(function (endpoint) {
      return '<a class="side-link" href="/monitoring/' + encodeURIComponent(endpoint.id) + '">' +
        '<span class="side-icon" aria-hidden="true">·</span>' +
        '<span class="side-label">' + esc(endpoint.name) + '</span></a>';
    }).join('');
  }


  /* ------------------------------------------------------------------ */
  /* KPI cards                                                           */
  /* ------------------------------------------------------------------ */

  function kpiCard(metric, label, icon, tone, format, invert) {
    var value = metric && metric.value !== undefined ? metric.value : 0;
    var previous = metric ? metric.previous : null;
    var delta = metric ? metric.delta_pct : null;
    var deltaHtml;
    if (delta !== null && delta !== undefined && isFinite(delta)) {
      var isUp = delta >= 0;
      var good = invert ? !isUp : isUp;
      var klass = delta === 0 ? 'is-flat' : good ? 'is-up' : 'is-down';
      deltaHtml = '<div class="kpi-delta ' + klass + '" title="Periodo anterior: ' +
        esc(format(previous)) + '">' + (isUp ? '▲' : '▼') + ' ' + esc(percent(Math.abs(delta), 1)) +
        ' vs. anterior</div>';
    } else {
      deltaHtml = '<div class="kpi-delta is-flat">' +
        (previous === null ? 'estado actual' : 'sin periodo previo') + '</div>';
    }
    return '<div class="kpi"><span class="kpi-icon' + (tone ? ' is-' + tone : '') + '" aria-hidden="true">' +
      icon + '</span><div class="kpi-body"><div class="kpi-value">' + esc(format(value)) + '</div>' +
      '<div class="kpi-label">' + esc(label) + '</div>' + deltaHtml + '</div></div>';
  }

  /* ------------------------------------------------------------------ */
  /* Global dashboard                                                    */
  /* ------------------------------------------------------------------ */

  function renderGlobalView(data, eventsResponse) {
    var kpis = data.kpis || {};
    if (dom.globalKpis) {
      dom.globalKpis.innerHTML =
        kpiCard(kpis.deployed_models, 'Modelos desplegados', '▦', 'blue', fmt) +
        kpiCard(kpis.active_endpoints, 'Endpoints activos', '●', 'green', fmt) +
        kpiCard(kpis.total_requests, 'Inferencias totales', '∑', '', fmt) +
        kpiCard(kpis.successful_requests, 'Correctas', '✓', 'green', fmt) +
        kpiCard(kpis.failed_requests, 'Fallidas', '✕', 'lava', fmt, true) +
        kpiCard(kpis.avg_latency_ms, 'Latencia media', '◷', '', ms, true) +
        kpiCard(kpis.error_rate, 'Tasa de error', '⚠', 'lava', function (value) { return percent(value, 2); }, true) +
        kpiCard(kpis.requests_per_minute, 'Peticiones / minuto', '↗', 'blue', fmt);
    }
    var range = data.range || {};
    if (byId('activity-sub')) {
      byId('activity-sub').textContent = range.label + ' · intervalos de ' + (range.bucket_label || '—');
    }
    var points = (data.series && data.series.activity) || [];
    LMCharts.stackedBars(byId('chart-activity'), {
      points: points,
      series: [
        { key: 'successful', label: 'Correctas', color: 'green' },
        { key: 'failed', label: 'Fallidas', color: 'lava' }
      ],
      height: 260,
      emptyText: 'Sin inferencias en el periodo seleccionado.',
      formatValue: fmt,
      onSelect: zoomToBucket
    });
    LMCharts.line(byId('chart-latency'), {
      points: points,
      series: [
        { key: 'avg_latency_ms', label: 'Media', color: 'blue', area: true },
        { key: 'p95_latency_ms', label: 'P95', color: 'amber', dashed: true },
        { key: 'p99_latency_ms', label: 'P99', color: 'lava', dashed: true }
      ],
      height: 260,
      formatValue: ms,
      formatAxis: function (value) { return ms(value); },
      emptyText: 'Sin latencias registradas todavía.',
      onSelect: zoomToBucket
    });
    LMCharts.line(byId('chart-error-rate'), {
      points: points,
      series: [{ key: 'error_rate', label: 'Tasa de error', color: 'lava', area: true, areaOpacity: 0.16 }],
      height: 220,
      formatValue: function (value) { return percent(value, 2); },
      formatAxis: function (value) { return value + '%'; },
      emptyText: 'Sin errores registrados en el periodo.',
      onSelect: zoomToBucket
    });


    renderPercentiles(data.totals, 'global-latency-percentiles');
    var breakdown = data.breakdown || {};
    var endpoints = breakdown.endpoints || [];
    LMCharts.bars(byId('chart-endpoints'), {
      items: endpoints.map(function (endpoint) {
        return {
          label: endpoint.endpoint_name,
          value: endpoint.requests,
          key: endpoint.endpoint_id,
          color: 'blue',
          meta: ms(endpoint.avg_latency_ms) + ' · ' + percent(endpoint.error_rate, 1) + ' de error'
        };
      }),
      limit: 8,
      formatValue: fmt,
      emptyText: 'Sin peticiones por endpoint en el periodo.',
      onSelect: function (item) { window.location.href = '/monitoring/' + encodeURIComponent(item.key); }
    });
    var latency = breakdown.latency || [];
    LMCharts.bars(byId('chart-latency-compare'), {
      items: latency.map(function (row) {
        return {
          label: row.endpoint_name,
          value: row.p95_latency_ms,
          key: row.endpoint_id,
          color: 'amber',
          meta: 'P50 ' + ms(row.p50_latency_ms) + ' · media ' + ms(row.avg_latency_ms)
        };
      }),
      limit: 8,
      formatValue: ms,
      emptyText: 'Sin latencias por endpoint en el periodo.',
      onSelect: function (item) { window.location.href = '/monitoring/' + encodeURIComponent(item.key); }
    });
    var models = breakdown.models || [];
    LMCharts.bars(byId('chart-models'), {
      items: models.map(function (row) {
        return {
          label: row.model_name,
          value: row.requests,
          key: row.model_id,
          color: 'navy',
          meta: row.endpoints + ' endpoints · ' + ms(row.avg_latency_ms)
        };
      }),
      limit: 8,
      formatValue: fmt,
      emptyText: 'Sin peticiones por modelo en el periodo.',
      onSelect: function (item) {
        if (dom.modelFilter) { dom.modelFilter.value = item.key; state.model = item.key; loadMonitoring(false); }
      }
    });
    var errors = breakdown.errors || {};
    var errorColors = ['lava', 'amber', 'blue', 'navy', 'muted'];
    LMCharts.donut(byId('chart-error-donut'), {
      items: (errors.by_type || []).map(function (item, index) {
        return { label: item.label, value: item.value, color: errorColors[index % errorColors.length] };
      }),
      centerLabel: errors.total ? 'errores' : 'sin errores',
      formatValue: fmt,
      emptyText: 'Sin errores en el periodo.',
      onSelect: function () {
        if (byId('event-status')) byId('event-status').value = 'error';
        state.eventStatus = 'error';
        loadEvents();
      }
    });
    renderTopErrors(errors, 'top-errors');
    renderEndpointHealth(endpoints, (data.options && data.options.endpoints) || []);
    renderResources(data.resources, 'resources-panel');
    renderEventHistory(eventsResponse.events || [], 'global-events', true);
  }


  /* ------------------------------------------------------------------ */
  /* Endpoint dashboard                                                  */
  /* ------------------------------------------------------------------ */

  function renderEndpointView(detail) {
    var endpoint = detail.endpoint || {};
    var deployment = detail.deployment;
    var status = deployment ? deployment.status : endpoint.status || 'deleted';
    var label = deployment ? deployment.status_label : 'Eliminado';
    var header = byId('endpoint-monitoring-header');
    if (header) {
      header.innerHTML = '<div class="page-header">' +
        '<div class="page-header-main">' +
        '<div class="page-eyebrow"><a href="/monitoring">Monitoring</a> · Endpoint</div>' +
        '<h1 class="page-title"><span class="page-icon" aria-hidden="true">◫</span> ' + esc(endpoint.name) +
        ' <span class="status-badge status-' + esc(status) + '"><span class="dot"></span>' + esc(label) + '</span></h1>' +
        '<p class="page-subtitle">' + esc(endpoint.model_name || endpoint.model_id || '') +
        (endpoint.method ? ' · <code>' + esc(endpoint.method) + '</code>' : '') +
        ' · estadísticas del endpoint en ' + esc((detail.range && detail.range.label) || '') + '.</p></div>' +
        '<div class="page-actions">' +
        '<a class="btn btn-secondary" href="/monitoring">← Overview global</a>' +
        (deployment ? '<a class="btn btn-primary" href="/serving/' + encodeURIComponent(deployment.id) + '">Abrir endpoint</a>' : '') +
        '</div></div>' +
        (deployment ? '' : '<div class="info-strip is-warn" style="margin-bottom:14px">Este endpoint ya no existe en Model Serving; se muestran los datos históricos del registro de inferencias.</div>');
    }
    if (dom.crumbCurrent) dom.crumbCurrent.textContent = endpoint.name || state.endpointRoute;
    var kpis = detail.kpis || {};
    if (dom.endpointKpis) {
      dom.endpointKpis.innerHTML =
        kpiCard(kpis.total_requests, 'Inferencias', '∑', 'blue', fmt) +
        kpiCard(kpis.successful_requests, 'Correctas', '✓', 'green', fmt) +
        kpiCard(kpis.failed_requests, 'Fallidas', '✕', 'lava', fmt, true) +
        kpiCard(kpis.error_rate, 'Tasa de error', '⚠', 'lava', function (value) { return percent(value, 2); }, true) +
        kpiCard(kpis.avg_latency_ms, 'Latencia media', '◷', '', ms, true) +
        kpiCard(kpis.p95_latency_ms, 'Latencia P95', '◷', 'lava', ms, true) +
        kpiCard(kpis.requests_per_minute, 'Peticiones / minuto', '↗', 'blue', fmt);
    }
    var points = (detail.series && detail.series.activity) || [];
    LMCharts.stackedBars(byId('endpoint-chart-activity'), {
      points: points,
      series: [
        { key: 'successful', label: 'Correctas', color: 'green' },
        { key: 'failed', label: 'Fallidas', color: 'lava' }
      ],
      height: 260,
      formatValue: fmt,
      emptyText: 'Sin peticiones para este endpoint en el periodo.',
      onSelect: zoomToBucket
    });
    LMCharts.line(byId('endpoint-chart-latency'), {
      points: points,
      series: [
        { key: 'avg_latency_ms', label: 'Media', color: 'blue', area: true },
        { key: 'p95_latency_ms', label: 'P95', color: 'amber', dashed: true },
        { key: 'p99_latency_ms', label: 'P99', color: 'lava', dashed: true }
      ],
      height: 260,
      formatValue: ms,
      formatAxis: function (value) { return ms(value); },
      emptyText: 'Sin latencias para este endpoint.',
      onSelect: zoomToBucket
    });
    var errors = detail.errors || {};
    LMCharts.donut(byId('endpoint-chart-errors'), {
      items: (errors.by_type || []).map(function (item, index) {
        return { label: item.label, value: item.value, color: ['lava', 'amber', 'blue', 'navy'][index % 4] };
      }),
      centerLabel: errors.total ? 'errores' : 'sin errores',
      formatValue: fmt,
      emptyText: 'Sin errores registrados.',
      size: 150
    });
    var usage = detail.usage || {};
    LMCharts.bars(byId('endpoint-chart-hours'), {
      items: (usage.hours || []).map(function (item) {
        return { label: item.label, value: item.value, color: 'blue' };
      }),
      limit: 24,
      formatValue: fmt,
      emptyText: 'Sin patrón de uso todavía.'
    });
    renderPercentiles(detail.totals, 'endpoint-latency-percentiles');
    renderTopErrors(errors, 'endpoint-top-errors');
    renderResources(detail.resources, 'endpoint-resources');
    renderEventHistory(detail.recent || [], 'endpoint-events', false);
    if (byId('endpoint-history-sub')) {
      byId('endpoint-history-sub').textContent = (detail.recent || []).length + ' peticiones mostradas · ' +
        ((detail.meta && detail.meta.data_source) || '');
    }
  }


  /* ------------------------------------------------------------------ */
  /* Secondary panels                                                    */
  /* ------------------------------------------------------------------ */

  function renderPercentiles(totals, targetId) {
    var box = byId(targetId);
    if (!box) return;
    var values = totals || {};
    box.innerHTML = [
      ['P50', values.p50_latency_ms],
      ['P90', values.p90_latency_ms],
      ['P95', values.p95_latency_ms],
      ['P99', values.p99_latency_ms],
      ['Máx.', values.max_latency_ms]
    ].map(function (entry) {
      return '<span class="latency-percentile"><span class="label">' + entry[0] + '</span>' +
        '<span class="value">' + esc(ms(entry[1] || 0)) + '</span></span>';
    }).join('');
  }

  function renderTopErrors(errors, targetId) {
    var box = byId(targetId);
    if (!box) return;
    var messages = (errors && errors.top_messages) || [];
    if (!messages.length) {
      box.innerHTML = emptyState('✓', 'Sin errores', 'No se han registrado inferencias fallidas en el periodo seleccionado.');
      return;
    }
    box.innerHTML = '<div class="error-list">' + messages.map(function (error) {
      return '<div class="error-item"><span class="error-count">' + esc(error.count) + '</span>' +
        '<div class="error-body"><div class="error-message">' + esc(error.message) + '</div>' +
        '<div class="error-meta">' + esc(error.error_type || 'Error') + ' · HTTP ' + esc(error.http_status || '—') +
        ' · ' + esc((error.endpoints || []).join(', ')) + ' · ' + esc(shortDate(error.last_at)) + '</div></div></div>';
    }).join('') + '</div>';
  }

  function statusLabel(status) {
    return {
      running: 'En ejecución',
      deploying: 'Desplegando',
      stopped: 'Detenido',
      failed: 'Fallido',
      deleted: 'Eliminado'
    }[status] || status || '—';
  }

  function renderEndpointHealth(endpoints, options) {
    var box = byId('endpoint-health');
    if (!box) return;
    var statuses = {};
    (options || []).forEach(function (option) { statuses[option.id] = option; });
    if (!endpoints.length) {
      box.innerHTML = emptyState('⇢', 'Sin tráfico', 'Ningún endpoint recibió peticiones en el periodo.');
      return;
    }
    box.innerHTML = '<div class="table-wrap"><table class="table"><caption class="sr-only">Salud por endpoint</caption>' +
      '<thead><tr><th>Endpoint</th><th>Estado</th><th class="num-head">Peticiones</th><th class="num-head">Fallidas</th><th class="num-head">Tasa error</th><th class="num-head">P95</th></tr></thead><tbody>' +
      endpoints.map(function (endpoint) {
        var option = statuses[endpoint.endpoint_id] || {};
        var status = option.status || 'deleted';
        return '<tr class="event-row">' +
          '<td><div class="cell-name"><span class="model-icon">·</span><span>' +
          '<a class="model-title" href="/monitoring/' + encodeURIComponent(endpoint.endpoint_id) + '">' +
          esc(endpoint.endpoint_name) + '</a><span class="model-id mono">' + esc(endpoint.endpoint_id) + '</span></span></div></td>' +
          '<td><span class="status-badge status-' + esc(status) + '"><span class="dot"></span>' +
          esc(statusLabel(status)) + '</span></td>' +
          '<td class="cell-num">' + fmt(endpoint.requests) + '</td>' +
          '<td class="cell-num">' + fmt(endpoint.failed) + '</td>' +
          '<td class="cell-num">' + percent(endpoint.error_rate, 1) + '</td>' +
          '<td class="cell-num">' + ms(endpoint.p95_latency_ms) + '</td></tr>';
      }).join('') + '</tbody></table></div>';
  }

  function renderResources(resources, targetId) {
    var box = byId(targetId);
    if (!box) return;
    if (!resources) {
      box.innerHTML = emptyState('◫', 'Sin muestra de recursos', 'El backend no devolvió métricas de recursos.');
      return;
    }
    if (!resources.available) {
      box.innerHTML = '<div class="info-strip is-warn" style="margin:4px 0"><strong>Recursos no disponibles.</strong> ' +
        esc(resources.reason || '') + '</div>' +
        '<p class="meta-line" style="margin-top:8px">' + esc(resources.note || '') + '</p>' +
        '<p class="meta-line" style="margin-top:6px">GPU: ' + esc((resources.gpu && resources.gpu.reason) || 'sin datos') + '</p>';
      return;
    }
    function gauge(label, value, suffix, hint) {
      return '<div style="margin-bottom:10px"><div class="lm-hbar-head"><span class="lm-hbar-label">' + esc(label) +
        '</span><span class="lm-hbar-value">' + esc(value) + esc(suffix || '') + '</span></div>' +
        '<div class="lm-hbar-track"><div class="lm-hbar-fill" style="width:' +
        Math.max(1, Math.min(100, Number(value) || 0)) + '%;background:var(--db-blue)"></div></div>' +
        (hint ? '<div class="lm-hbar-meta">' + esc(hint) + '</div>' : '') + '</div>';
    }
    box.innerHTML = gauge('CPU del proceso', resources.cpu_percent, '%') +
      gauge('Memoria del proceso', resources.memory_percent, '%',
        (resources.memory_mb || 0) + ' MB') +
      gauge('CPU del sistema', resources.system_cpu_percent, '%') +
      gauge('Memoria del sistema', resources.system_memory_percent, '%') +
      '<p class="meta-line" style="margin-top:4px">Procesos hijos activos: ' + esc(resources.children || 0) +
      '. ' + esc(resources.note || '') + '</p>';
  }


  function renderEventHistory(events, targetId, withEndpoint) {
    var box = byId(targetId);
    if (!box) return;
    if (!events.length) {
      box.innerHTML = emptyState('◷', 'Sin peticiones', 'El historial aparecerá cuando se invoque un endpoint en el periodo seleccionado.');
      return;
    }
    box.innerHTML = '<div class="table-wrap"><table class="table"><caption class="sr-only">Historial de peticiones</caption>' +
      '<thead><tr><th>Fecha</th>' + (withEndpoint ? '<th>Endpoint</th>' : '') +
      '<th>Estado</th><th class="num-head">HTTP</th><th class="num-head">Latencia</th><th>Error</th></tr></thead><tbody>' +
      events.map(function (event) {
        return '<tr class="event-row" data-monitor-event="' + esc(event.id) + '">' +
          '<td class="mono" style="font-size:10.5px;white-space:nowrap">' + esc(isoDate(event.at)) + '</td>' +
          (withEndpoint ? '<td><a href="/monitoring/' + encodeURIComponent(event.endpoint_id) + '">' +
            esc(event.endpoint_name || event.endpoint_id) + '</a></td>' : '') +
          '<td><span class="event-status ' + (event.status === 'success' ? 'is-ok' : 'is-error') + '">' +
          (event.status === 'success' ? '✓ Correcta' : '✕ Fallida') + '</span></td>' +
          '<td class="cell-num">' + esc(event.http_status || '—') + '</td>' +
          '<td class="cell-num">' + ms(event.latency_ms || 0) + '</td>' +
          '<td class="event-error" title="' + esc(event.error_message || '') + '">' + esc(event.error_message || '') + '</td></tr>';
      }).join('') + '</tbody></table></div>';
    box.onclick = function (event) {
      var row = event.target.closest('tr[data-monitor-event]');
      if (!row) return;
      var found = events.filter(function (item) { return item.id === row.getAttribute('data-monitor-event'); })[0];
      if (found) renderEventDetail(found, targetId === 'endpoint-events' ? 'endpoint-event-detail' : 'global-event-detail');
    };
  }

  function renderEventDetail(event, targetId) {
    var box = byId(targetId);
    if (!box) return;
    box.hidden = false;
    box.innerHTML = '<div class="panel-body" style="border-top:1px solid var(--db-line-soft)">' +
      '<div class="review-grid">' +
      review('Fecha', isoDate(event.at)) +
      review('Endpoint', event.endpoint_name || event.endpoint_id || '—') +
      review('Modelo', event.model_name || event.model_id || '—') +
      review('Método', event.method || '—') +
      review('Estado', (event.status || '') + ' · HTTP ' + (event.http_status || '—')) +
      review('Latencia', ms(event.latency_ms || 0)) +
      '</div>' +
      '<div class="code-head" style="margin-top:12px"><span class="code-label">Entrada</span></div>' +
      '<pre class="code-highlight">' + esc(event.input_preview || '—') + '</pre>' +
      (event.output_preview ? '<div class="code-head" style="margin-top:8px"><span class="code-label">Salida</span></div>' +
        '<pre class="code-highlight">' + esc(event.output_preview) + '</pre>' : '') +
      (event.error_message ? '<div class="wizard-alert is-error" style="margin-top:8px"><strong>' +
        esc(event.error_type || 'Error') + '</strong><br>' + esc(event.error_message) + '</div>' : '') +
      (event.error_detail ? '<div class="code-head" style="margin-top:8px"><span class="code-label">Traza</span></div>' +
        '<pre class="code-highlight">' + esc(event.error_detail) + '</pre>' : '') +
      '</div>';
  }

  function review(label, value) {
    return '<div class="review-card"><div class="label">' + esc(label) + '</div><div class="value">' +
      esc(value) + '</div></div>';
  }


  /* ------------------------------------------------------------------ */
  /* Interactions                                                        */
  /* ------------------------------------------------------------------ */

  function loadEvents() {
    if (!dom.events) return;
    api('/api/monitoring/events?' + queryString({
      limit: 50,
      status: state.eventStatus || ''
    })).then(function (data) {
      renderEventHistory(data.events || [], 'global-events', true);
      state.events = data.events || [];
    }).catch(function (error) {
      dom.events.innerHTML = emptyState('⚠', 'No se pudo cargar el historial', error.message);
    });
  }

  function localInputValue(epochSeconds) {
    var date = new Date(epochSeconds * 1000);
    var pad = function (value) { return String(value).padStart(2, '0'); };
    return date.getFullYear() + '-' + pad(date.getMonth() + 1) + '-' + pad(date.getDate()) +
      'T' + pad(date.getHours()) + ':' + pad(date.getMinutes());
  }

  function toIso(value) {
    if (!value) return '';
    var date = new Date(value);
    return isNaN(date.getTime()) ? '' : date.toISOString();
  }

  function zoomToBucket(point) {
    var source = state.data || state.detail;
    if (!point || !source || !point.t) return;
    var bucket = (source.range && source.range.bucket_seconds) || 3600;
    state.range = 'custom';
    state.customStart = localInputValue(point.t);
    state.customEnd = localInputValue(point.t + bucket);
    if (byId('range-start')) byId('range-start').value = state.customStart;
    if (byId('range-end')) byId('range-end').value = state.customEnd;
    if (byId('custom-range')) byId('custom-range').hidden = false;
    syncRangeButtons();
    applyCustomRange();
    toast('Rango ajustado al intervalo seleccionado');
  }

  function syncRangeButtons() {
    each(document.querySelectorAll('#range-buttons button'), function (button) {
      button.setAttribute('aria-pressed', button.getAttribute('data-range') === state.range ? 'true' : 'false');
    });
    if (byId('custom-range')) byId('custom-range').hidden = state.range !== 'custom';
  }

  function selectRange(range) {
    state.range = range;
    syncRangeButtons();
    if (range === 'custom') {
      if (byId('range-start') && !byId('range-start').value) {
        var span = (state.data && state.data.range) || (state.detail && state.detail.range) || {};
        var end = span.end || Date.now() / 1000;
        var start = span.start || end - 86400;
        byId('range-start').value = localInputValue(start);
        byId('range-end').value = localInputValue(end);
        state.customStart = toIso(byId('range-start').value);
        state.customEnd = toIso(byId('range-end').value);
      }
      return;
    }
    loadMonitoring(false);
  }

  function applyCustomRange() {
    var startInput = byId('range-start');
    var endInput = byId('range-end');
    var start = startInput ? startInput.value : '';
    var end = endInput ? endInput.value : '';
    if (!start) { toast('Indica la fecha de inicio', true); return; }
    if (!end) end = localInputValue(Date.now() / 1000);
    var startIso = toIso(start);
    var endIso = toIso(end);
    if (!startIso || !endIso) { toast('Las fechas no son válidas', true); return; }
    if (new Date(startIso) >= new Date(endIso)) {
      toast('La fecha de inicio debe ser anterior a la de fin', true);
      return;
    }
    state.range = 'custom';
    state.customStart = startIso;
    state.customEnd = endIso;
    syncRangeButtons();
    loadMonitoring(false);
  }

  function scheduleAutoRefresh() {
    if (state.timer) window.clearTimeout(state.timer);
    state.timer = 0;
    if (!state.auto) return;
    state.timer = window.setTimeout(function () {
      if (!document.hidden) loadMonitoring(false);
      scheduleAutoRefresh();
    }, 30000);
  }

  function bindControls() {
    each(document.querySelectorAll('#range-buttons button'), function (button) {
      button.addEventListener('click', function () {
        selectRange(button.getAttribute('data-range'));
      });
    });
    var apply = byId('apply-custom-range');
    if (apply) apply.addEventListener('click', applyCustomRange);
    if (dom.modelFilter) dom.modelFilter.addEventListener('change', function () {
      state.model = dom.modelFilter.value;
      loadMonitoring(false);
    });
    if (dom.endpointFilter) dom.endpointFilter.addEventListener('change', function () {
      state.endpoint = dom.endpointFilter.value;
      loadMonitoring(false);
    });
    if (dom.events) dom.events.addEventListener('change', function () {});
    var statusFilter = byId('event-status');
    if (statusFilter) statusFilter.addEventListener('change', function () {
      state.eventStatus = statusFilter.value;
      loadEvents();
    });
    var refresh = byId('refresh-monitoring');
    if (refresh) refresh.addEventListener('click', function () { loadMonitoring(false); });
    var refreshEvents = byId('refresh-events');
    if (refreshEvents) refreshEvents.addEventListener('click', loadEvents);
    var auto = byId('auto-refresh');
    if (auto) auto.addEventListener('change', function () {
      state.auto = auto.checked;
      toast(state.auto ? 'Actualización automática activada' : 'Actualización automática desactivada');
      scheduleAutoRefresh();
    });
    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) scheduleAutoRefresh();
    });
  }


  /* ------------------------------------------------------------------ */
  /* Startup                                                             */
  /* ------------------------------------------------------------------ */

  function boot() {
    dom.toast = byId('toast');
    dom.serverStatus = byId('server-status');
    dom.serverStatusText = byId('server-status-text');
    dom.sideVersion = byId('side-version');
    dom.loading = byId('monitoring-loading');
    dom.error = byId('monitoring-error');
    dom.globalView = byId('monitoring-global');
    dom.endpointView = byId('monitoring-endpoint-view');
    dom.globalKpis = byId('global-kpis');
    dom.endpointKpis = byId('endpoint-kpis');
    dom.modelFilter = byId('monitor-model');
    dom.endpointFilter = byId('monitor-endpoint');
    dom.events = byId('global-events');
    dom.updated = byId('monitor-updated');
    dom.crumbCurrent = byId('crumb-current');

    var match = window.location.pathname.match(/^\/monitoring\/([^/]+)\/?$/);
    state.endpointRoute = match && match[1] ? decodeURIComponent(match[1]) : '';
    if (state.endpointRoute) {
      if (byId('model-filter-group')) byId('model-filter-group').hidden = true;
      if (byId('endpoint-filter-group')) byId('endpoint-filter-group').hidden = true;
      if (byId('side-overview')) byId('side-overview').classList.remove('is-active');
      if (byId('monitoring-header')) byId('monitoring-header').hidden = true;
    }

    bindSidebar();
    bindTheme();
    bindControls();
    syncRangeButtons();
    loadHealth();
    if (state.endpointRoute) {
      api('/api/serving/endpoints').then(function (data) {
        state.sidebarEndpoints = (data.endpoints || []).map(function (endpoint) {
          return {
            id: endpoint.id,
            name: endpoint.name,
            status: endpoint.status,
            exists: true
          };
        });
        updateSidebarEndpoints();
      }).catch(function () { /* the dashboard still works */ });
    }
    loadMonitoring(true);
    scheduleAutoRefresh();
  }

  document.addEventListener('DOMContentLoaded', boot);
})();


