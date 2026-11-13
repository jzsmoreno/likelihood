'use strict';

/*
 * Model Serving workspace.
 *
 * Every view is driven by the real API: the model picker comes from the catalog,
 * the methods and samples from the introspection endpoints, the endpoint lifecycle
 * from /api/serving and the request history from the serving event log.
 */
(function () {
  var API = '/api';
  var state = {
    health: null,
    models: [],
    instances: [],
    reuseInstance: null,
    modelFilter: '',
    modelCategory: '',
    selectedModel: null,
    modelDetail: null,
    invokeSpec: null,
    playground: null,
    methods: [],
    method: null,
    loadEnabled: false,
    loader: '',
    loadPath: '',
    parametersText: '{}',
    step: 1,
    nameTouched: false,
    endpoints: [],
    endpointsLoaded: false,
    endpoint: null,
    docs: null,
    events: [],
    detailTab: 'overview',
    pollTimer: 0,
    filePath: ''
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

  function api(path, options) {
    var config = options || {};
    var headers = { Accept: 'application/json' };
    if (config.body !== undefined) headers['Content-Type'] = 'application/json';
    return fetch(path, {
      method: config.method || 'GET',
      headers: headers,
      body: config.body === undefined ? undefined : JSON.stringify(config.body),
      credentials: 'same-origin'
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

  function copyText(value, okMessage) {
    var text = String(value === null || value === undefined ? '' : value);
    if (!text.trim()) { toast('No hay nada que copiar', true); return; }
    var done = function () { toast(okMessage || 'Copiado'); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { legacyCopy(text, done); });
      return;
    }
    legacyCopy(text, done);
  }

  function legacyCopy(value, onDone) {
    var area = document.createElement('textarea');
    area.value = value;
    area.setAttribute('readonly', 'readonly');
    area.style.position = 'fixed';
    area.style.top = '-1000px';
    document.body.appendChild(area);
    try { area.select(); document.execCommand('copy'); } catch (error) { /* ignore */ }
    document.body.removeChild(area);
    onDone();
  }

  function fmt(value) { return window.LMCharts ? LMCharts.fmtNumber(value) : String(value); }
  function fmtMs(value) { return window.LMCharts ? LMCharts.fmtMs(value) : (value + ' ms'); }
  function pct(value) { return window.LMCharts ? LMCharts.fmtPercent(value) : (value + '%'); }

  function isoDate(value) {
    if (!value) return '—';
    var date = new Date(value);
    return isNaN(date.getTime()) ? String(value) : date.toLocaleString('es-ES');
  }

  function slug(value) {
    return String(value || '')
      .normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
      .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'endpoint';
  }

  function statusBadge(status, label) {
    var text = label || ({ running: 'En ejecución', deploying: 'Desplegando', stopped: 'Detenido', failed: 'Fallido', deleted: 'Eliminado' }[status] || status);
    return '<span class="status-badge status-' + esc(status) + '"><span class="dot"></span>' + esc(text) + '</span>';
  }

  function emptyState(icon, title, text, action) {
    return '<div class="empty-state"><span class="empty-icon" aria-hidden="true">' + icon + '</span>' +
      '<h3>' + esc(title) + '</h3><p>' + esc(text) + '</p>' + (action || '') + '</div>';
  }

  function setStatus(node, message, kind) {
    if (!node) return;
    node.textContent = message || '';
    node.className = 'status-note' + (kind ? ' ' + kind : '');
  }

  /* ------------------------------------------------------------------ */
  /* Shell: sidebar, theme, service status and routing                   */
  /* ------------------------------------------------------------------ */

  function isMobileLayout() {
    return window.matchMedia('(max-width: 900px)').matches;
  }

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
    var match = window.location.pathname.match(/^\/serving\/([^/]+)\/?$/);
    if (match && match[1]) showDetail(decodeURIComponent(match[1]));
    else showList();
  }

  function showList() {
    stopPolling();
    state.endpoint = null;
    if (dom.listView) dom.listView.hidden = false;
    if (dom.detailView) dom.detailView.hidden = true;
    if (dom.crumbCurrent) dom.crumbCurrent.textContent = 'Endpoints';
    loadEndpoints();
    loadHealth();
  }

  function showDetail(endpointId) {
    if (dom.listView) dom.listView.hidden = true;
    if (dom.detailView) dom.detailView.hidden = false;
    if (dom.crumbCurrent) dom.crumbCurrent.textContent = endpointId;
    if (dom.detail) dom.detail.innerHTML = '<div class="loader"><span class="spinner"></span> Cargando endpoint…</div>';
    loadEndpoint(endpointId);
    loadHealth();
  }

  function stopPolling() {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = 0;
  }

  /* ------------------------------------------------------------------ */
  /* Endpoint list                                                       */
  /* ------------------------------------------------------------------ */

  function loadEndpoints() {
    if (!dom.results) return;
    dom.results.setAttribute('aria-busy', 'true');
    return api('/api/serving/endpoints').then(function (data) {
      state.endpoints = data.endpoints || [];
      state.endpointsLoaded = true;
      renderKpis();
      populateModelFilter();
      renderEndpoints();
      if (dom.countEndpoints) dom.countEndpoints.textContent = String(state.endpoints.length);
    }).catch(function (error) {
      state.endpointsLoaded = true;
      if (dom.kpis) dom.kpis.innerHTML = '';
      dom.results.innerHTML = emptyState(
        '⚠', 'No se pudieron cargar los endpoints', error.message,
        '<button class="btn btn-secondary btn-sm" type="button" data-action="refresh">Reintentar</button>'
      );
      if (dom.countEndpoints) dom.countEndpoints.textContent = '—';
    }).then(function () {
      dom.results.setAttribute('aria-busy', 'false');
    });
  }

  function renderKpis() {
    if (!dom.kpis) return;
    var counts = { running: 0, deploying: 0, requests: 0, errors: 0, latency: 0, withRequests: 0 };
    state.endpoints.forEach(function (endpoint) {
      if (endpoint.status === 'running') counts.running += 1;
      if (endpoint.status === 'deploying') counts.deploying += 1;
      var stats = endpoint.stats || {};
      counts.requests += Number(stats.requests || 0);
      counts.errors += Number(stats.failed || 0);
      if (stats.requests) {
        counts.latency += Number(stats.avg_latency_ms || 0) * Number(stats.requests);
        counts.withRequests += Number(stats.requests);
      }
    });
    var avg = counts.withRequests ? counts.latency / counts.withRequests : 0;
    var errorRate = counts.requests ? counts.errors / counts.requests * 100 : 0;
    dom.kpis.innerHTML = [
      kpi('⇢', state.endpoints.length, 'Endpoints desplegados', 'blue'),
      kpi('●', counts.running, 'Activos', 'green', counts.deploying ? counts.deploying + ' desplegando' : ''),
      kpi('∑', fmt(counts.requests), 'Inferencias totales', '', counts.errors ? fmt(counts.errors) + ' fallidas' : ''),
      kpi('◷', fmtMs(avg), 'Latencia media', 'lava', counts.requests ? pct(errorRate) + ' de error' : '')
    ].join('');
  }

  function kpi(icon, value, label, tone, note) {
    return '<div class="kpi"><span class="kpi-icon' + (tone ? ' is-' + tone : '') + '" aria-hidden="true">' +
      icon + '</span><div class="kpi-body"><div class="kpi-value">' + esc(value) + '</div>' +
      '<div class="kpi-label">' + esc(label) + '</div>' +
      (note ? '<div class="kpi-delta">' + esc(note) + '</div>' : '') + '</div></div>';
  }

  function populateModelFilter() {
    if (!dom.modelFilter) return;
    var selected = dom.modelFilter.value;
    var models = {};
    state.endpoints.forEach(function (endpoint) {
      models[endpoint.model_id] =
        (endpoint.model_title || endpoint.model_name || endpoint.model_type_id || endpoint.model_id) +
        ' · ' + endpoint.model_id;
    });
    var html = '<option value="">Todos</option>';
    Object.keys(models).sort().forEach(function (key) {
      html += '<option value="' + esc(key) + '">' + esc(models[key]) + '</option>';
    });
    dom.modelFilter.innerHTML = html;
    dom.modelFilter.value = selected;
  }

  function filteredEndpoints() {
    var term = (dom.search ? dom.search.value : '').trim().toLowerCase();
    var status = dom.statusFilter ? dom.statusFilter.value : '';
    var model = dom.modelFilter ? dom.modelFilter.value : '';
    return state.endpoints.filter(function (endpoint) {
      if (status && endpoint.status !== status) return false;
      if (model && endpoint.model_id !== model) return false;
      if (!term) return true;
      return [endpoint.name, endpoint.model_id, endpoint.model_name, endpoint.method, endpoint.description]
        .join(' ').toLowerCase().indexOf(term) !== -1;
    });
  }


  function renderEndpoints() {
    if (!dom.results) return;
    var rows = filteredEndpoints();
    if (!state.endpoints.length) {
      dom.results.innerHTML = emptyState(
        '⇢',
        'Todavía no hay endpoints',
        'Despliega un método de inferencia del catálogo para obtener una URL lista para consumir, ejemplos de API y métricas reales.',
        '<button class="btn btn-primary btn-sm" type="button" data-action="create">＋ Crear endpoint</button>'
      );
    } else if (!rows.length) {
      dom.results.innerHTML = emptyState(
        '⌕', 'Ningún endpoint coincide', 'Ajusta la búsqueda o los filtros para ver otros despliegues.',
        '<button class="btn btn-secondary btn-sm" type="button" data-action="clear-filters">Restablecer filtros</button>'
      );
    } else {
      dom.results.innerHTML = '<div class="table-wrap"><table class="table">' +
        '<caption class="sr-only">Endpoints desplegados</caption><thead><tr>' +
        '<th>Endpoint</th><th>Modelo</th><th>Método</th><th>Estado</th>' +
        '<th class="num-head">Inferencias</th><th class="num-head">Errores</th>' +
        '<th class="num-head">Latencia</th><th class="cell-actions">Acciones</th></tr></thead><tbody>' +
        rows.map(endpointRow).join('') + '</tbody></table></div>';
    }
    if (dom.resultBar) dom.resultBar.hidden = !state.endpoints.length;
    if (dom.resultCount) {
      dom.resultCount.textContent = rows.length + ' de ' + state.endpoints.length +
        (state.endpoints.length === 1 ? ' endpoint' : ' endpoints');
    }
    if (dom.resultHint) {
      var active = state.endpoints.filter(function (item) { return item.status === 'running'; }).length;
      dom.resultHint.textContent = active + ' activo' + (active === 1 ? '' : 's');
    }
  }

  function endpointRow(endpoint) {
    var stats = endpoint.stats || {};
    var errorRate = Number(stats.error_rate || 0);
    return '<tr data-endpoint="' + esc(endpoint.id) + '">' +
      '<td><div class="cell-name"><span class="model-icon">' + esc(endpoint.model_icon || '⇢') + '</span>' +
      '<span><a class="model-title" href="/serving/' + encodeURIComponent(endpoint.id) + '">' + esc(endpoint.name) + '</a>' +
      '<span class="model-id mono">' + esc(endpoint.id) + '</span></span></div></td>' +
      '<td><a href="/preview/' + encodeURIComponent(endpoint.model_type_id || endpoint.model_id) + '">' +
      esc(endpoint.model_title || endpoint.model_name) + '</a>' +
      '<span class="model-id mono">' + esc(endpoint.model_type_id || endpoint.model_id) + '</span>' +
      '<span class="model-id mono" title="model_id de la instancia">' + esc(endpoint.model_id) + '</span></td>' +
      '<td><span class="mono">' + esc(endpoint.method) + '</span></td>' +
      '<td>' + statusBadge(endpoint.status, endpoint.status_label) + '</td>' +
      '<td class="cell-num">' + fmt(stats.requests || 0) + '</td>' +
      '<td class="cell-num">' + (errorRate ? pct(errorRate) : '0%') + '</td>' +
      '<td class="cell-num">' + fmtMs(stats.avg_latency_ms || 0) + '</td>' +
      '<td class="cell-actions">' +
      '<a class="btn btn-ghost btn-sm" href="/serving/' + encodeURIComponent(endpoint.id) + '">Abrir</a>' +
      (endpoint.status === 'running' ? '<button class="btn btn-ghost btn-sm" type="button" data-action="invoke" data-endpoint="' + esc(endpoint.id) + '">Probar</button>' : '') +
      '</td></tr>';
  }

  function bindEndpointTable() {
    if (!dom.results) return;
    dom.results.addEventListener('click', function (event) {
      var button = event.target.closest('button[data-action]');
      if (!button) return;
      var action = button.getAttribute('data-action');
      if (action === 'refresh') loadEndpoints();
      if (action === 'create') openWizard();
      if (action === 'clear-filters') {
        if (dom.search) dom.search.value = '';
        if (dom.statusFilter) dom.statusFilter.value = '';
        if (dom.modelFilter) dom.modelFilter.value = '';
        renderEndpoints();
      }
      if (action === 'invoke') {
        window.location.href = '/serving/' + encodeURIComponent(button.getAttribute('data-endpoint')) + '#probar';
      }
    });
  }


  /* ------------------------------------------------------------------ */
  /* Deployment wizard                                                   */
  /* ------------------------------------------------------------------ */

  function defaultParameters() {
    var values = {};
    var fields = (state.playground && state.playground.fields) || [];
    fields.forEach(function (field) {
      var value = field.default;
      if (value === null || value === undefined) value = field.sample;
      if (value !== null && value !== undefined) values[field.name] = value;
    });
    return JSON.stringify(values, null, 2);
  }

  function preferredMethod(methods) {
    var preferred = ['predict', 'predict_proba', 'forward', 'transform', 'call', 'infer', 'score'];
    for (var index = 0; index < preferred.length; index += 1) {
      var found = (methods || []).filter(function (method) { return method.name === preferred[index]; })[0];
      if (found) return found;
    }
    return (methods || [])[0] || null;
  }

  function resetWizard(modelId) {
    state.selectedModel = modelId ? { id: modelId } : null;
    state.reuseInstance = null;
    state.modelFilter = '';
    state.modelCategory = '';
    state.modelDetail = null;
    state.invokeSpec = null;
    state.playground = null;
    state.methods = [];
    state.method = null;
    state.loadEnabled = false;
    state.loader = '';
    state.loadPath = '';
    state.parametersText = '{}';
    state.step = 1;
    state.nameTouched = false;
    state.name = '';
    state.description = '';
    state.timeout = 60;
    state.concurrency = 1;
  }

  function openWizard(modelId) {
    resetWizard(modelId);
    var catalogRequest = state.models.length
      ? Promise.resolve(state.models)
      : api('/api/models?limit=200').then(function (data) {
          state.models = data.models || [];
          return state.models;
        });
    Promise.all([
      catalogRequest,
      api('/api/serving/models').then(function (data) {
        state.instances = data.models || [];
        return state.instances;
      }).catch(function () { return []; })
    ]).then(function () {
      showWizard();
      if (modelId) selectModel(modelId);
    }).catch(function (error) {
      toast(error.message, true);
    });
  }

  function showWizard() {
    if (!dom.deployDialog) return;
    dom.deployDialog.hidden = false;
    window.requestAnimationFrame(function () { dom.deployDialog.classList.add('is-open'); });
    document.body.classList.add('db-body-locked');
    renderWizard();
  }

  function closeWizard() {
    if (!dom.deployDialog) return;
    dom.deployDialog.classList.remove('is-open');
    window.setTimeout(function () {
      dom.deployDialog.hidden = true;
      document.body.classList.remove('db-body-locked');
    }, 160);
  }

  function setStep(step) {
    if (step > 1 && !state.selectedModel) return;
    if (step > 2 && !state.method) return;
    state.step = step;
    renderWizard();
  }

  function renderWizard() {
    if (!dom.deployDialog || dom.deployDialog.hidden) return;
    each(document.querySelectorAll('#wizard-steps .wizard-step'), function (button) {
      var index = Number(button.getAttribute('data-step'));
      button.classList.toggle('is-active', index === state.step);
      button.classList.toggle('is-done', index < state.step);
      button.setAttribute('aria-selected', index === state.step ? 'true' : 'false');
      button.disabled = index === 1 ? false : index === 2 ? !state.selectedModel : index === 3 ? !state.method : !state.method;
    });
    each(document.querySelectorAll('.wizard-panel'), function (panel) {
      panel.hidden = panel.id !== 'wizard-panel-' + state.step;
    });
    var renderers = { 1: renderStepModel, 2: renderStepLoad, 3: renderStepMethod, 4: renderStepConfig };
    renderers[state.step]();
    if (dom.wizardBack) dom.wizardBack.disabled = state.step === 1;
    if (dom.wizardNext) {
      dom.wizardNext.textContent = state.step === 4 ? 'Desplegar endpoint' : 'Continuar';
      dom.wizardNext.disabled = false;
    }
    setStatus(dom.wizardStatus, '');
  }

  function captureWizard() {
    if (state.step === 1) {
      var search = byId('wizard-model-search');
      if (search) state.modelFilter = search.value;
    }
    if (state.step === 2) {
      var enabled = byId('wizard-load-enabled');
      if (enabled) state.loadEnabled = enabled.checked;
      var loader = byId('wizard-loader');
      if (loader) state.loader = loader.value;
      var path = byId('wizard-load-path');
      if (path) state.loadPath = path.value;
      var parameters = byId('wizard-parameters');
      if (parameters) state.parametersText = parameters.value;
    }
    if (state.step === 4) {
      var name = byId('wizard-name');
      if (name) { state.name = name.value; }
      var description = byId('wizard-description');
      if (description) state.description = description.value;
      var timeout = byId('wizard-timeout');
      if (timeout) state.timeout = Number(timeout.value || 60);
      var concurrency = byId('wizard-concurrency');
      if (concurrency) state.concurrency = Number(concurrency.value || 1);
    }
  }

  function nextStep() {
    captureWizard();
    if (state.step === 1) {
      if (!state.selectedModel) { setStatus(dom.wizardStatus, 'Selecciona un modelo para continuar', 'is-error'); return; }
      if (!state.method) { setStatus(dom.wizardStatus, 'Selecciona un método de inferencia', 'is-error'); return; }
    }
    if (state.step === 2) {
      if (state.loadEnabled && !state.loadPath.trim()) {
        setStatus(dom.wizardStatus, 'Indica la ruta del modelo guardado', 'is-error'); return;
      }
    }
    if (state.step < 4) { setStep(state.step + 1); return; }
    deployEndpoint();
  }


  function selectModel(modelId, instance) {
    var brief = state.models.filter(function (model) { return model.id === modelId; })[0] || { id: modelId };
    if (instance) {
      state.reuseInstance = instance;
    } else if (!state.reuseInstance || state.reuseInstance.model_type_id !== modelId) {
      state.reuseInstance = null;
    }
    if (state.selectedModel && state.selectedModel.id !== modelId) {
      state.loadEnabled = false;
      state.loadPath = '';
      state.loader = '';
    }
    state.selectedModel = brief;
    setStatus(dom.wizardStatus, 'Cargando el modelo…', 'is-busy');
    Promise.all([
      api('/api/models/' + encodeURIComponent(modelId)),
      api('/api/models/' + encodeURIComponent(modelId) + '/invoke'),
      api('/api/playground/' + encodeURIComponent(modelId))
    ]).then(function (results) {
      state.modelDetail = results[0].model;
      state.selectedModel = state.modelDetail;
      state.invokeSpec = results[1];
      state.playground = results[2].playground;
      state.methods = (results[1].methods || []).filter(function (method) {
        return !/^(fit|train|partial_fit|compile|build|save|load|plot|render|viz|freeze|unfreeze|set_|get_config|get_params)/i
          .test(method.name || '');
      });
      state.method = preferredMethod(state.methods);
      state.parametersText = defaultParameters();
      if (state.reuseInstance) {
        state.parametersText = JSON.stringify(state.reuseInstance.parameters || {}, null, 2);
        state.loadEnabled = Boolean(
          state.reuseInstance.load && state.reuseInstance.load.enabled
        );
        state.loader = (state.reuseInstance.load && state.reuseInstance.load.loader) || '';
        state.loadPath = (state.reuseInstance.load && state.reuseInstance.load.path) || '';
      }
      state.nameTouched = false;
      if (state.method) {
        state.name = slug(modelId) + '-' + slug(state.method.name);
        state.timeout = results[1].timeout_seconds || 60;
      }
      setStatus(dom.wizardStatus, '');
      renderWizard();
    }).catch(function (error) {
      setStatus(dom.wizardStatus, error.message, 'is-error');
    });
  }

  function renderStepModel() {
    var panel = byId('wizard-panel-1');
    if (!panel) return;
    var term = (state.modelFilter || '').trim().toLowerCase();
    var categories = {};
    state.models.forEach(function (model) {
      categories[model.category] = model.category_label || model.category;
    });
    var models = state.models.filter(function (model) {
      if (state.modelCategory && model.category !== state.modelCategory) return false;
      if (!term) return true;
      return [model.id, model.title, model.category_label, model.module, model.short_description]
        .join(' ').toLowerCase().indexOf(term) !== -1;
    });
    var categoryOptions = '<option value="">Todas las categorías</option>' +
      Object.keys(categories).sort().map(function (key) {
        return '<option value="' + esc(key) + '"' + (state.modelCategory === key ? ' selected' : '') + '>' +
          esc(categories[key]) + '</option>';
      }).join('');
    var instanceOptions = '<option value="">Crear una instancia nueva</option>' +
      state.instances.map(function (instance) {
        var selected = state.reuseInstance && state.reuseInstance.model_id === instance.model_id;
        var label = (instance.name || instance.model_name || instance.model_id) +
          ' · ' + instance.model_type_id + ' · ' + instance.model_id;
        return '<option value="' + esc(instance.model_id) + '"' + (selected ? ' selected' : '') + '>' +
          esc(label) + '</option>';
      }).join('');
    panel.innerHTML = '<h3 class="wizard-section-title">1 · Selecciona un modelo</h3>' +
      '<p class="wizard-section-sub">Elige el tipo de modelo del catálogo o reutiliza una instancia registrada. Cada instancia tiene un <code>model_id</code> único; un modelo puede alimentar varios deployments.</p>' +
      '<div class="wizard-field" style="margin-bottom:12px"><label for="wizard-model-instance">Instancia de modelo (opcional)</label>' +
      '<select class="select" id="wizard-model-instance">' + instanceOptions + '</select>' +
      '<span class="hint">Reutilizar una instancia conserva sus parámetros y su artefacto cargado.</span></div>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">' +
      '<div class="search-field" style="max-width:none;flex:1 1 220px">' +
      '<span class="search-icon" aria-hidden="true"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="7" cy="7" r="4.4"/><path d="M10.3 10.3 14 14" stroke-linecap="round"/></svg></span>' +
      '<label class="sr-only" for="wizard-model-search">Buscar modelo</label>' +
      '<input type="search" id="wizard-model-search" placeholder="Buscar por nombre, categoría, módulo…" value="' + esc(state.modelFilter) + '" autocomplete="off">' +
      '</div>' +
      '<select class="select" id="wizard-model-category" aria-label="Filtrar por categoría">' + categoryOptions + '</select>' +
      '</div>' +
      '<div class="model-picker" id="wizard-model-list">' +
      (models.length ? models.map(modelPick).join('') : '<p class="empty">No hay modelos que coincidan con la búsqueda.</p>') +
      '</div>' + modelSummary();
    var search = byId('wizard-model-search');
    if (search) {
      search.addEventListener('input', function () {
        state.modelFilter = search.value;
        renderStepModel();
        var next = byId('wizard-model-search');
        if (next) { next.focus(); next.setSelectionRange(next.value.length, next.value.length); }
      });
    }
    var category = byId('wizard-model-category');
    if (category) {
      category.addEventListener('change', function () {
        state.modelCategory = category.value;
        renderStepModel();
      });
    }
    var instanceSelect = byId('wizard-model-instance');
    if (instanceSelect) {
      instanceSelect.addEventListener('change', function () {
        var found = state.instances.filter(function (item) {
          return item.model_id === instanceSelect.value;
        })[0];
        if (found) {
          state.reuseInstance = found;
          selectModel(found.model_type_id, found);
        } else {
          state.reuseInstance = null;
          state.parametersText = defaultParameters();
          state.loadEnabled = false;
          state.loadPath = '';
          state.loader = '';
          renderWizard();
        }
      });
    }
    each(panel.querySelectorAll('.model-pick'), function (button) {
      button.addEventListener('click', function () {
        state.reuseInstance = null;
        selectModel(button.getAttribute('data-model'));
      });
    });
  }

  function modelPick(model) {
    var selected = state.selectedModel && state.selectedModel.id === model.id;
    return '<button class="model-pick' + (selected ? ' is-selected' : '') + '" type="button" data-model="' + esc(model.id) + '">' +
      '<span class="model-pick-icon">' + esc(model.icon || '📦') + '</span>' +
      '<span class="model-pick-body"><span class="model-pick-title">' + esc(model.title || model.id) +
      '<span class="badge">' + esc(model.category_label || model.category || '') + '</span>' +
      (model.loadable ? '<span class="badge badge-green">Cargable</span>' : '') + '</span>' +
      '<span class="model-pick-desc">' + esc(model.short_description || model.signature || '') + '</span></span>' +
      '<span class="model-pick-meta">' + esc(model.method_count || 0) + ' métodos<br>' + esc(model.module || '') + '</span>' +
      '</button>';
  }

  function modelSummary() {
    if (!state.selectedModel || !state.modelDetail) return '';
    var model = state.modelDetail;
    return '<div class="model-summary">' +
      summaryItem('Modelo', model.name) +
      (state.reuseInstance
        ? summaryItem('Instancia reutilizada', state.reuseInstance.model_id)
        : summaryItem('Nueva instancia', 'se generará un model_id único')) +
      summaryItem('Versión', (state.health && state.health.version ? 'v' + state.health.version : '—') +
        ' · ' + (model.module || '')) +
      summaryItem('Tipo', 'Clase · ' + (model.category_label || model.category)) +
      summaryItem('Capacidades', (model.loaders && model.loaders.length ? 'Cargable' : 'Construible') + ' · ' + state.methods.length + ' métodos invocables') +
      summaryItem('Descripción', model.short_description) +
      summaryItem('Firma', model.signature) +
      '</div>';
  }

  function summaryItem(label, value) {
    return '<div class="summary-item"><div class="label">' + esc(label) + '</div><div class="value">' +
      esc(value || '—') + '</div></div>';
  }


  function renderStepLoad() {
    var panel = byId('wizard-panel-2');
    if (!panel || !state.modelDetail) return;
    if (state.reuseInstance) {
      var instance = state.reuseInstance;
      panel.innerHTML = '<h3 class="wizard-section-title">2 · Modelo reutilizado</h3>' +
        '<p class="wizard-section-sub">Este deployment usará una instancia de modelo ya registrada. No se crea otro modelo ni se modifica el existente.</p>' +
        '<div class="info-strip"><strong>model_id:</strong> <code>' + esc(instance.model_id) +
        '</code> · <strong>tipo:</strong> <code>' + esc(instance.model_type_id) + '</code></div>' +
        '<div class="review-grid" style="margin-top:12px">' +
        reviewCard('Instancia', instance.name || instance.model_id) +
        reviewCard('Fuente', instance.load && instance.load.enabled
          ? 'Modelo guardado · ' + (instance.load.loader || '') + ' · ' + (instance.load.path || '')
          : 'Constructor') +
        reviewCard('Parámetros', JSON.stringify(instance.parameters || {})) +
        reviewCard('Deployments', String(instance.deployment_count || 0)) +
        '</div>';
      return;
    }
    var model = state.modelDetail;
    var loaders = model.loaders || [];
    var canLoad = loaders.length > 0;
    if (!state.loader && canLoad) state.loader = loaders[0].name;
    var loaderOptions = loaders.map(function (loader) {
      return '<option value="' + esc(loader.name) + '"' + (state.loader === loader.name ? ' selected' : '') + '>' +
        esc(loader.signature || loader.name) + '</option>';
    }).join('');
    panel.innerHTML = '<h3 class="wizard-section-title">2 · Configura la carga del modelo</h3>' +
      '<p class="wizard-section-sub">Puedes desplegar una instancia construida desde el constructor o cargar un artefacto ya entrenado. La carga se valida durante el despliegue con el mismo backend del catálogo.</p>' +
      (canLoad
        ? '<label class="check-row" for="wizard-load-enabled"><input type="checkbox" id="wizard-load-enabled"' +
          (state.loadEnabled ? ' checked' : '') + '><span><strong>Cargar un modelo entrenado antes de desplegar</strong>' +
          '<span>Se usarán los loaders reales del catálogo y la política de rutas del servidor (LM_LOAD_ROOT).</span></span></label>'
        : '<div class="info-strip is-warn">Este modelo no declara un método <code>load()</code>: se construirá con los parámetros del constructor en cada inferencia.</div>') +
      '<div id="wizard-load-fields" style="margin-top:12px"' + (state.loadEnabled && canLoad ? '' : ' hidden') + '>' +
      '<div class="wizard-fields">' +
      '<div class="wizard-field"><label for="wizard-loader">Loader</label>' +
      '<select class="select" id="wizard-loader">' + loaderOptions + '</select></div>' +
      '<div class="wizard-field"><label for="wizard-load-path">Ruta del modelo guardado</label>' +
      '<div style="display:flex;gap:6px"><input class="input" type="text" id="wizard-load-path" value="' + esc(state.loadPath) + '" placeholder="output/model.pkl">' +
      '<button class="btn btn-secondary" type="button" id="wizard-browse">Examinar…</button></div>' +
      '<span class="hint">Relativa a LM_LOAD_ROOT o absoluta.</span></div>' +
      '</div></div>' +
      '<div class="wizard-field" style="margin-top:14px"><label for="wizard-parameters">Parámetros del constructor (JSON)</label>' +
      '<textarea class="wizard-json" id="wizard-parameters" spellcheck="false">' + esc(state.parametersText) + '</textarea>' +
      '<span class="hint">Valores reales del catálogo. Los loaders estáticos los ignoran; un loader que necesita instancia los usa para construirla.</span></div>' +
      '<div class="form-actions"><button class="btn btn-secondary btn-sm" type="button" id="wizard-reset-parameters">Restablecer desde el catálogo</button></div>';

    var enabled = byId('wizard-load-enabled');
    if (enabled) {
      enabled.addEventListener('change', function () {
        state.loadEnabled = enabled.checked;
        var fields = byId('wizard-load-fields');
        if (fields) fields.hidden = !enabled.checked;
      });
    }
    var browse = byId('wizard-browse');
    if (browse) {
      browse.addEventListener('click', function () {
        openFileBrowser(function (path) {
          state.loadPath = path;
          var input = byId('wizard-load-path');
          if (input) input.value = path;
        });
      });
    }
    var reset = byId('wizard-reset-parameters');
    if (reset) {
      reset.addEventListener('click', function () {
        state.parametersText = defaultParameters();
        var area = byId('wizard-parameters');
        if (area) area.value = state.parametersText;
      });
    }
  }


  function renderStepMethod() {
    var panel = byId('wizard-panel-3');
    if (!panel) return;
    panel.innerHTML = '<h3 class="wizard-section-title">3 · Selecciona el método de inferencia</h3>' +
      '<p class="wizard-section-sub">Solo aparecen métodos invocables y compatibles con JSON. La firma y los argumentos provienen de la introspección real.</p>' +
      '<div class="method-list">' +
      (state.methods.length
        ? state.methods.map(methodPick).join('')
        : '<p class="empty">Este modelo no expone métodos invocables con argumentos JSON.</p>') +
      '</div>' + methodDetail();
    each(panel.querySelectorAll('.method-pick'), function (button) {
      button.addEventListener('click', function () {
        var name = button.getAttribute('data-method');
        state.method = state.methods.filter(function (method) { return method.name === name; })[0] || null;
        if (state.method && !state.nameTouched) {
          state.name = slug(state.modelDetail.id) + '-' + slug(state.method.name);
        }
        renderWizard();
      });
    });
  }

  function looksInference(name) {
    return /(predict|infer|forecast|score|transform|forward|call|sample|generate|encode|decode|simulate|step|act|probability|anomaly)/i.test(name || '');
  }

  function methodPick(method) {
    var selected = state.method && state.method.name === method.name;
    return '<button class="method-pick' + (selected ? ' is-selected' : '') + '" type="button" data-method="' + esc(method.name) + '">' +
      '<span class="method-pick-body"><span class="method-pick-name">' + esc(method.name) +
      (looksInference(method.name) ? ' <span class="badge badge-green">inferencia</span>' : '') + '</span>' +
      '<span class="method-pick-signature">' + esc(method.signature) + '</span>' +
      '<span class="method-pick-desc">' + esc(method.description || 'Sin descripción') + '</span></span>' +
      '</button>';
  }

  function methodDetail() {
    if (!state.method) return '';
    var fields = state.method.fields || [];
    return '<div class="panel" style="margin-top:12px"><div class="panel-header"><h3>Contrato del método</h3>' +
      '<span class="mono" style="font-size:10.5px;color:var(--db-faint)">' + esc(state.method.signature) + '</span></div>' +
      '<div class="panel-body" style="padding:0"><table class="schema-table"><thead><tr><th>Argumento</th><th>Tipo</th><th>Obligatorio</th><th>Ejemplo</th></tr></thead><tbody>' +
      (fields.length
        ? fields.map(function (field) {
            return '<tr><td class="mono">' + esc(field.name) + '</td><td class="mono">' + esc(field.type || 'Any') + '</td>' +
              '<td>' + (field.required ? '<span class="schema-required">Sí</span>' : 'No') + '</td>' +
              '<td class="mono">' + esc(JSON.stringify(field.sample)) + '</td></tr>';
          }).join('')
        : '<tr><td colspan="4">El método no declara argumentos.</td></tr>') +
      '</tbody></table></div></div>' +
      '<div class="code-head" style="margin-top:12px"><span class="code-label">Ejemplo de payload</span>' +
      '<button class="btn btn-ghost btn-xs" type="button" id="copy-method-example">Copiar</button></div>' +
      '<pre class="code-highlight" id="method-example">' + esc(JSON.stringify(exampleForMethod(state.method), null, 2)) + '</pre>';
  }

  function exampleForMethod(method) {
    if (!method) return {};
    var fields = method.fields || [];
    if (fields.length === 1) return { input: fields[0].sample === null ? 'YOUR_INPUT' : fields[0].sample };
    var payload = {};
    fields.forEach(function (field) {
      if (field.sample !== null && field.sample !== undefined) payload[field.name] = field.sample;
    });
    return { inputs: payload };
  }


  function renderStepConfig() {
    var panel = byId('wizard-panel-4');
    if (!panel || !state.method) return;
    var loadSummary = state.loadEnabled
      ? 'Cargar ' + state.loadPath + ' con ' + state.loader + '()'
      : 'Construir con el constructor';
    panel.innerHTML = '<h3 class="wizard-section-title">4 · Configura y despliega</h3>' +
      '<p class="wizard-section-sub">El endpoint se valida antes de crearse. La carga del artefacto se ejecuta en segundo plano y el estado pasa de <b>deploying</b> a <b>running</b>.</p>' +
      '<div class="wizard-fields">' +
      '<div class="wizard-field"><label for="wizard-name">Nombre del endpoint</label>' +
      '<input class="input" type="text" id="wizard-name" value="' + esc(state.name) + '" pattern="[A-Za-z0-9][A-Za-z0-9._-]{1,62}">' +
      '<span class="hint">2–63 caracteres: letras, números, punto, guion y guion bajo.</span></div>' +
      '<div class="wizard-field"><label for="wizard-timeout">Timeout por petición (s)</label>' +
      '<input class="input" type="number" id="wizard-timeout" min="1" max="3600" step="1" value="' + esc(state.timeout) + '"></div>' +
      '<div class="wizard-field"><label for="wizard-concurrency">Concurrencia máxima</label>' +
      '<input class="input" type="number" id="wizard-concurrency" min="1" max="32" step="1" value="' + esc(state.concurrency) + '">' +
      '<span class="hint">Peticiones simultáneas antes de responder 429.</span></div>' +
      '<div class="wizard-field"><label for="wizard-description">Descripción</label>' +
      '<input class="input" type="text" id="wizard-description" value="' + esc(state.description) + '" placeholder="Endpoint de inferencia para…"></div>' +
      '</div>' +
      '<div class="review-grid" style="margin-top:14px">' +
      reviewCard('Modelo', state.selectedModel.name + ' · ' + (state.selectedModel.signature || '')) +
      reviewCard('Model ID', state.reuseInstance ? state.reuseInstance.model_id : 'nuevo · se generará al desplegar') +
      reviewCard('Deployment ID', 'se generará desde el nombre del endpoint') +
      reviewCard('Método', state.method.name + ' · ' + state.method.signature) +
      reviewCard('Fuente', loadSummary) +
      reviewCard('Runtime', state.timeout + ' s · ' + state.concurrency + ' en paralelo') +
      '</div>' +
      '<div class="form-actions"><button class="btn btn-secondary btn-sm" type="button" id="wizard-validate">Validar configuración</button></div>' +
      '<div id="wizard-validation" hidden></div>';

    var name = byId('wizard-name');
    if (name) name.addEventListener('input', function () { state.name = name.value; state.nameTouched = true; });
    var description = byId('wizard-description');
    if (description) description.addEventListener('input', function () { state.description = description.value; });
    var timeout = byId('wizard-timeout');
    if (timeout) timeout.addEventListener('input', function () { state.timeout = Number(timeout.value || 60); });
    var concurrency = byId('wizard-concurrency');
    if (concurrency) concurrency.addEventListener('input', function () { state.concurrency = Number(concurrency.value || 1); });
    var validate = byId('wizard-validate');
    if (validate) validate.addEventListener('click', function () { validateDeployment(true); });
  }

  function reviewCard(label, value) {
    return '<div class="review-card"><div class="label">' + esc(label) + '</div><div class="value">' +
      esc(value || '—') + '</div></div>';
  }

  function wizardPayload() {
    var parameters;
    try {
      parameters = JSON.parse(state.parametersText || '{}');
    } catch (error) {
      throw new Error('Los parámetros del constructor no son JSON válido: ' + error.message);
    }
    if (parameters === null || typeof parameters !== 'object' || Array.isArray(parameters)) {
      throw new Error('Los parámetros del constructor deben ser un objeto JSON');
    }
    return {
      model_id: state.reuseInstance ? state.reuseInstance.model_id : undefined,
      model_type_id: state.reuseInstance
        ? state.reuseInstance.model_type_id
        : (state.modelDetail.model_type_id || state.modelDetail.id),
      method: state.method.name,
      endpoint_name: state.name,
      description: state.description,
      parameters: parameters,
      config: {
        timeout_seconds: Number(state.timeout || 60),
        max_concurrency: Number(state.concurrency || 1)
      },
      load: {
        enabled: Boolean(state.loadEnabled && state.loadPath.trim()),
        loader: state.loadEnabled ? state.loader : null,
        path: state.loadEnabled ? state.loadPath.trim() : null
      }
    };
  }


  function showValidation(result, container) {
    var box = container || byId('wizard-validation');
    if (!box) return;
    box.hidden = false;
    var warnings = (result.warnings || []).map(function (item) { return '<li>' + esc(item) + '</li>'; }).join('');
    if (!result.valid) {
      box.innerHTML = '<div class="wizard-alert is-error"><strong>La configuración no es válida.</strong><ul>' +
        (result.errors || [result.error || 'Error desconocido']).map(function (item) {
          return '<li>' + esc(item) + '</li>';
        }).join('') + '</ul></div>';
      return;
    }
    box.innerHTML = '<div class="wizard-alert' + (warnings ? ' is-warn' : '') + '">' +
      '<strong>Configuración válida.</strong> El endpoint se llamará <code>' + esc(result.name) + '</code>.' +
      (warnings ? '<ul>' + warnings + '</ul>' : '') + '</div>';
  }

  function validateDeployment(show) {
    if (!state.modelDetail || !state.method) return Promise.resolve(false);
    var payload;
    try {
      payload = wizardPayload();
    } catch (error) {
      setStatus(dom.wizardStatus, error.message, 'is-error');
      if (show) showValidation({ valid: false, errors: [error.message] });
      return Promise.resolve(false);
    }
    return api('/api/serving/validate', { method: 'POST', body: payload }).then(function (result) {
      if (show) showValidation(result);
      return Boolean(result.valid);
    }).catch(function (error) {
      if (show) showValidation({ valid: false, errors: [error.message] });
      setStatus(dom.wizardStatus, error.message, 'is-error');
      return false;
    });
  }

  function deployEndpoint() {
    captureWizard();
    var payload;
    try {
      payload = wizardPayload();
    } catch (error) {
      setStatus(dom.wizardStatus, error.message, 'is-error');
      return;
    }
    if (dom.wizardNext) dom.wizardNext.disabled = true;
    setStatus(dom.wizardStatus, 'Validando la configuración…', 'is-busy');
    api('/api/serving/validate', { method: 'POST', body: payload }).then(function (result) {
      if (!result.valid) {
        showValidation(result);
        if (dom.wizardNext) dom.wizardNext.disabled = false;
        setStatus(dom.wizardStatus, 'Revisa la configuración marcada', 'is-error');
        return null;
      }
      setStatus(dom.wizardStatus, 'Creando el endpoint…', 'is-busy');
      return api('/api/serving/endpoints', { method: 'POST', body: payload });
    }).then(function (result) {
      if (!result) return;
      var endpoint = result.endpoint || result;
      toast('Endpoint creado; el despliegue continúa en segundo plano');
      closeWizard();
      window.location.href = '/serving/' + encodeURIComponent(endpoint.id);
    }).catch(function (error) {
      var details = (error.data && error.data.invalid) || [];
      showValidation({ valid: false, errors: [error.message].concat(details) });
      setStatus(dom.wizardStatus, error.message, 'is-error');
      if (dom.wizardNext) dom.wizardNext.disabled = false;
    });
  }


  /* ------------------------------------------------------------------ */
  /* Endpoint detail                                                     */
  /* ------------------------------------------------------------------ */

  function loadEndpoint(endpointId) {
    stopPolling();
    return Promise.all([
      api('/api/serving/endpoints/' + encodeURIComponent(endpointId)),
      api('/api/serving/endpoints/' + encodeURIComponent(endpointId) + '/events?limit=100')
    ]).then(function (results) {
      state.endpoint = results[0].endpoint || results[0];
      state.events = results[1].events || [];
      renderDetail();
      if (state.endpoint.status === 'deploying') pollEndpoint(endpointId);
    }).catch(function (error) {
      if (dom.detail) {
        dom.detail.innerHTML = emptyState(
          '⚠', 'Endpoint no disponible', error.message,
          '<a class="btn btn-secondary btn-sm" href="/serving">Volver a los endpoints</a>'
        );
      }
    });
  }

  function pollEndpoint(endpointId) {
    stopPolling();
    state.pollTimer = window.setTimeout(function () {
      api('/api/serving/endpoints/' + encodeURIComponent(endpointId)).then(function (data) {
        var endpoint = data.endpoint || data;
        var previous = state.endpoint ? state.endpoint.status : '';
        state.endpoint = endpoint;
        renderDetail();
        if (endpoint.status === 'deploying') pollEndpoint(endpointId);
        else if (previous === 'deploying') {
          toast(endpoint.status === 'running' ? 'Endpoint desplegado correctamente' : 'El despliegue falló', endpoint.status !== 'running');
        }
      }).catch(function () { pollEndpoint(endpointId); });
    }, 1500);
  }

  function renderDetail() {
    var endpoint = state.endpoint;
    if (!dom.detail || !endpoint) return;
    var hash = (window.location.hash || '').replace('#', '');
    if (['overview', 'probar', 'api', 'actividad'].indexOf(hash) !== -1) state.detailTab = hash;
    if (dom.crumbCurrent) dom.crumbCurrent.textContent = endpoint.name;
    dom.detail.innerHTML = heroHtml(endpoint) + tabsHtml();
    renderTab();
  }

  function heroHtml(endpoint) {
    var running = endpoint.status === 'running';
    var deploying = endpoint.status === 'deploying';
    return '<div class="endpoint-hero">' +
      '<div class="endpoint-hero-main">' +
      '<span class="endpoint-hero-icon" aria-hidden="true">' + esc(endpoint.model_icon || '⇢') + '</span>' +
      '<div class="endpoint-hero-body">' +
      '<div class="page-eyebrow"><a href="/serving">Endpoints</a> · Serving</div>' +
      '<h1>' + esc(endpoint.name) + statusBadge(endpoint.status, endpoint.status_label) + '</h1>' +
      '<div class="endpoint-hero-sub">' +
      '<span><a href="/preview/' + encodeURIComponent(endpoint.model_type_id || endpoint.model_id) + '">' +
      esc(endpoint.model_title || endpoint.model_name) + '</a></span>' +
      '<span class="mono">' + esc(endpoint.method_signature || endpoint.method) + '</span>' +
      '<span>revisión v' + esc(endpoint.revision || 1) + '</span>' +
      '<span class="mono">model_id: ' + esc(endpoint.model_id) + '</span>' +
      '</div></div>' +
      '<div class="endpoint-hero-actions">' +
      (running ? '<button class="btn btn-primary btn-sm" type="button" data-endpoint-action="test">Probar</button>' : '') +
      (running ? '<button class="btn btn-secondary btn-sm" type="button" data-endpoint-action="stop">Detener</button>' : '') +
      (!running && !deploying ? '<button class="btn btn-primary btn-sm" type="button" data-endpoint-action="restart">Reiniciar</button>' : '') +
      (deploying ? '<button class="btn btn-secondary btn-sm" type="button" disabled><span class="spinner" style="width:12px;height:12px"></span> Desplegando</button>' : '') +
      '<button class="btn btn-ghost btn-sm" type="button" data-endpoint-action="clear-metrics">Borrar métricas</button>' +
      '<button class="btn btn-ghost btn-sm" type="button" data-endpoint-action="delete">Eliminar</button>' +
      '</div></div>' +
      '<div class="endpoint-url-bar">' +
      '<span class="endpoint-url-label">Endpoint URL</span>' +
      '<span class="endpoint-url mono" id="endpoint-url" title="' + esc(endpoint.endpoint_url) + '">' + esc(endpoint.endpoint_url) + '</span>' +
      '<button class="btn btn-secondary btn-sm" type="button" id="copy-endpoint-url">Copiar URL</button>' +
      '<a class="btn btn-ghost btn-sm" href="/monitoring/' + encodeURIComponent(endpoint.id) + '">Monitorización</a>' +
      '</div>' +
      '<div class="endpoint-meta-grid">' +
      meta('Modelo', endpoint.model_type_id || endpoint.model_id) +
      meta('Model ID', endpoint.model_id) +
      meta('Deployment ID', endpoint.deployment_id || endpoint.id) +
      meta('Método', endpoint.method) +
      meta('Fuente', endpoint.load && endpoint.load.enabled ? 'Modelo cargado' : 'Constructor') +
      meta('Timeout', endpoint.config.timeout_seconds + ' s') +
      meta('Concurrencia', String(endpoint.config.max_concurrency)) +
      meta('Creado', isoDate(endpoint.created_at)) +
      meta('Autenticación', endpoint.auth_required ? 'Bearer token' : 'Abierta') +
      '</div>' +
      (endpoint.last_error ? '<div style="padding:0 16px 14px"><div class="info-strip is-error"><strong>Último error:</strong> ' + esc(endpoint.last_error) + '</div></div>' : '') +
      '</div>';
  }

  function meta(label, value) {
    return '<div class="endpoint-meta"><div class="endpoint-meta-label">' + esc(label) + '</div>' +
      '<div class="endpoint-meta-value mono" title="' + esc(value) + '">' + esc(value) + '</div></div>';
  }


  function tabsHtml() {
    var tabs = [
      ['overview', 'Resumen'],
      ['probar', 'Probar'],
      ['api', 'API'],
      ['actividad', 'Actividad']
    ];
    return '<div class="panel" style="overflow:hidden">' +
      '<div class="tabs" role="tablist" aria-label="Detalle del endpoint">' +
      tabs.map(function (tab) {
        return '<button class="tab" type="button" role="tab" data-detail-tab="' + tab[0] + '" aria-selected="' +
          (state.detailTab === tab[0] ? 'true' : 'false') + '">' + tab[1] + '</button>';
      }).join('') +
      '</div>' +
      '<div class="tab-panel" id="detail-panel-overview" ' + (state.detailTab === 'overview' ? '' : 'hidden') + '></div>' +
      '<div class="tab-panel" id="detail-panel-probar" ' + (state.detailTab === 'probar' ? '' : 'hidden') + '></div>' +
      '<div class="tab-panel" id="detail-panel-api" ' + (state.detailTab === 'api' ? '' : 'hidden') + '></div>' +
      '<div class="tab-panel" id="detail-panel-actividad" ' + (state.detailTab === 'actividad' ? '' : 'hidden') + '></div>' +
      '</div>';
  }

  function renderTab() {
    if (!state.endpoint) return;
    if (state.detailTab === 'overview') renderOverview();
    if (state.detailTab === 'probar') renderInvokePanel();
    if (state.detailTab === 'api') renderApiPanel();
    if (state.detailTab === 'actividad') renderActivity();
    each(document.querySelectorAll('[data-detail-tab]'), function (button) {
      button.setAttribute('aria-selected', button.getAttribute('data-detail-tab') === state.detailTab ? 'true' : 'false');
    });
  }

  function switchTab(tab) {
    state.detailTab = tab;
    each(document.querySelectorAll('.tab-panel'), function (panel) {
      panel.hidden = panel.id !== 'detail-panel-' + tab;
    });
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, '', window.location.pathname + '#' + tab);
    }
    renderTab();
  }

  function bindDetailActions() {
    var detail = dom.detail;
    if (!detail) return;
    detail.addEventListener('click', function (event) {
      var tab = event.target.closest('[data-detail-tab]');
      if (tab) { switchTab(tab.getAttribute('data-detail-tab')); return; }
      if (event.target.closest('#copy-endpoint-url')) {
        copyText(state.endpoint.endpoint_url, 'URL copiada');
        return;
      }
      var action = event.target.closest('[data-endpoint-action]');
      if (!action) return;
      var name = action.getAttribute('data-endpoint-action');
      if (name === 'test') switchTab('probar');
      if (name === 'stop') endpointAction('stop');
      if (name === 'restart') endpointAction('restart');
      if (name === 'clear-metrics') endpointAction('clear-metrics');
      if (name === 'delete') endpointAction('delete');
    });
  }

  function endpointAction(action) {
    var endpoint = state.endpoint;
    if (!endpoint) return;
    if (action === 'clear-metrics') {
      confirmAction(
        'Borrar métricas del deployment',
        'Se eliminarán todas las métricas de <strong>' + esc(endpoint.name) +
          '</strong> (deployment <code>' + esc(endpoint.deployment_id || endpoint.id) + '</code>). ' +
          'El deployment, su modelo (<code>' + esc(endpoint.model_id) + '</code>) y las métricas de otros deployments no se modifican.',
        function () {
          return api('/api/serving/endpoints/' + encodeURIComponent(endpoint.id) + '/metrics', {
            method: 'DELETE'
          }).then(function (result) {
            toast((result.metrics_deleted || 0) + ' métricas eliminadas');
            loadEndpoint(endpoint.id);
          }).catch(function (error) { toast(error.message, true); });
        }
      );
      return;
    }
    if (action === 'delete') {
      confirmAction(
        'Eliminar endpoint',
        'El endpoint <strong>' + esc(endpoint.name) + '</strong> dejará de aceptar peticiones. ' +
          'También se eliminarán automáticamente todas sus métricas de monitoring. ' +
          'El modelo del catálogo y la instancia <code>' + esc(endpoint.model_id) + '</code> se conservan.',
        function () {
          return api('/api/serving/endpoints/' + encodeURIComponent(endpoint.id), { method: 'DELETE' }).then(function (result) {
            toast('Endpoint eliminado · ' + (result.metrics_deleted || 0) + ' métricas borradas');
            window.location.href = '/serving';
          }).catch(function (error) { toast(error.message, true); });
        }
      );
      return;
    }
    if (action === 'stop') {
      confirmAction(
        'Detener endpoint',
        'Las peticiones a <strong>' + esc(endpoint.name) + '</strong> responderán 409 hasta que lo reinicies.',
        function () {
          return api('/api/serving/endpoints/' + encodeURIComponent(endpoint.id) + '/stop', { method: 'POST' }).then(function () {
            toast('Endpoint detenido');
            loadEndpoint(endpoint.id);
          }).catch(function (error) { toast(error.message, true); });
        }
      );
      return;
    }
    if (action === 'restart') {
      api('/api/serving/endpoints/' + encodeURIComponent(endpoint.id) + '/restart', { method: 'POST' }).then(function () {
        toast('Reiniciando el endpoint…');
        loadEndpoint(endpoint.id);
      }).catch(function (error) { toast(error.message, true); });
    }
  }


  function renderOverview() {
    var panel = byId('detail-panel-overview');
    if (!panel) return;
    var endpoint = state.endpoint;
    var stats = endpoint.stats || {};
    var load = endpoint.load || { enabled: false };
    var fields = endpoint.method_fields || [];
    panel.innerHTML =
      '<div class="kpi-grid is-dense" style="margin-bottom:14px">' +
      kpi('∑', fmt(stats.requests || 0), 'Inferencias totales', 'blue') +
      kpi('✓', fmt(stats.successful || 0), 'Correctas', 'green') +
      kpi('⚠', fmt(stats.failed || 0), 'Fallidas', 'lava', stats.requests ? pct(stats.error_rate || 0) + ' de error' : '') +
      kpi('◷', fmtMs(stats.avg_latency_ms || 0), 'Latencia media', '', stats.p95_latency_ms ? 'P95 ' + fmtMs(stats.p95_latency_ms) : '') +
      '</div>' +
      (endpoint.status === 'failed'
        ? '<div class="info-strip is-error" style="margin-bottom:14px"><strong>El despliegue falló.</strong> ' +
          esc(endpoint.last_error || 'Consulta el registro de despliegue.') + '</div>'
        : '') +
      '<div class="chart-grid is-2">' +
      '<section class="panel"><div class="panel-header"><h2>Configuración del endpoint</h2>' +
      '<span class="panel-sub">inmutable mientras está en ejecución</span></div><div class="panel-body">' +
      '<div class="review-grid">' +
      reviewCard('Endpoint', endpoint.name) +
      reviewCard('Deployment ID', endpoint.deployment_id || endpoint.id) +
      reviewCard('Modelo', endpoint.model_type_id || endpoint.model_id) +
      reviewCard('Model ID', endpoint.model_id) +
      reviewCard('Método', endpoint.method) +
      reviewCard('Timeout', endpoint.config.timeout_seconds + ' s') +
      reviewCard('Concurrencia', String(endpoint.config.max_concurrency)) +
      reviewCard('Autenticación', endpoint.auth_required ? 'Bearer token' : 'Abierta') +
      reviewCard('Creado', isoDate(endpoint.created_at)) +
      reviewCard('Desplegado', endpoint.deployed_at ? isoDate(endpoint.deployed_at) : '—') +
      '</div>' +
      (endpoint.description ? '<p style="margin-top:12px;color:var(--db-muted);font-size:12px">' + esc(endpoint.description) + '</p>' : '') +
      '</div></section>' +
      '<section class="panel"><div class="panel-header"><h2>Origen del modelo</h2>' +
      '<span class="panel-sub">' + (load.enabled ? 'modelo entrenado' : 'constructor') + '</span></div><div class="panel-body">' +
      (load.enabled
        ? '<div class="review-grid">' +
          reviewCard('Loader', load.loader) +
          reviewCard('Ruta', load.path) +
          reviewCard('Validado', endpoint.loaded_artifact ? isoDate(endpoint.loaded_artifact.loaded_at) : 'En curso') +
          reviewCard('Tipo cargado', endpoint.loaded_artifact ? (endpoint.loaded_artifact.type || '—') : '—') +
          '</div>' +
          (endpoint.loaded_artifact && endpoint.loaded_artifact.elapsed
            ? '<p class="kpi-label" style="margin-top:8px">La última carga validada tardó ' + esc(fmtMs(endpoint.loaded_artifact.elapsed * 1000)) + '.</p>'
            : '')
        : '<p style="color:var(--db-muted);font-size:12px">Cada inferencia construye el modelo con los parámetros del constructor. Si el modelo declara un loader, puedes crear otro endpoint cargando un artefacto entrenado.</p>') +
      '<div class="code-head" style="margin-top:12px"><span class="code-label">Parámetros del constructor</span>' +
      '<button class="btn btn-ghost btn-xs" type="button" data-copy-json="' + esc(JSON.stringify(endpoint.parameters || {})) + '">Copiar</button></div>' +
      '<pre class="code-highlight">' + esc(JSON.stringify(endpoint.parameters || {}, null, 2)) + '</pre>' +
      '</div></section>' +
      '</div>' +
      '<div class="chart-grid is-2" style="margin-top:0">' +
      '<section class="panel"><div class="panel-header"><h2>Registro de despliegue</h2>' +
      '<span class="panel-sub">' + (endpoint.status === 'deploying' ? 'actualizándose…' : 'últimos pasos') + '</span></div>' +
      '<div class="panel-body">' + deployLog(endpoint.log || []) + '</div></section>' +
      '<section class="panel"><div class="panel-header"><h2>Contrato del método</h2>' +
      '<span class="mono" style="font-size:10.5px;color:var(--db-faint)">' + esc(endpoint.method_signature || '') + '</span></div>' +
      '<div class="panel-body" style="padding:0"><table class="schema-table"><thead><tr><th>Argumento</th><th>Tipo</th><th>Obligatorio</th><th>Ejemplo</th></tr></thead><tbody>' +
      (fields.length
        ? fields.map(function (field) {
            return '<tr><td class="mono">' + esc(field.name) + '</td><td class="mono">' + esc(field.type || 'Any') + '</td>' +
              '<td>' + (field.required ? '<span class="schema-required">Sí</span>' : 'No') + '</td>' +
              '<td class="mono">' + esc(JSON.stringify(field.sample)) + '</td></tr>';
          }).join('')
        : '<tr><td colspan="4">El método no declara argumentos.</td></tr>') +
      '</tbody></table></div></section></div>';
  }

  function deployLog(entries) {
    if (!entries.length) return '<p class="empty">Sin eventos de despliegue registrados.</p>';
    return '<ul class="deploy-log">' + entries.map(function (entry) {
      return '<li class="is-' + esc(entry.level || 'info') + '"><span class="deploy-log-time">' +
        esc((entry.at || '').replace('T', ' ').replace('Z', '')) + '</span>' +
        '<span class="deploy-log-message">' + esc(entry.message) + '</span></li>';
    }).join('') + '</ul>';
  }


  /* ------------------------------------------------------------------ */
  /* Inference playground                                                */
  /* ------------------------------------------------------------------ */

  function renderInvokePanel() {
    var panel = byId('detail-panel-probar');
    if (!panel) return;
    var endpoint = state.endpoint;
    var fields = endpoint.method_fields || [];
    var example = endpoint.example_payload || {};
    var running = endpoint.status === 'running';
    panel.innerHTML =
      '<p class="form-note">La petición se envía al endpoint real y queda registrada en Model Monitoring. ' +
      (fields.length === 1
        ? 'El método declara un único argumento, así que puedes usar el atajo <code>{"input": …}</code>.'
        : 'Usa <code>{"inputs": {…}}</code> con los nombres reales de los argumentos.') +
      '</p>' +
      (running ? '' : '<div class="info-strip is-error" style="margin-bottom:12px">El endpoint está <b>' +
        esc(endpoint.status) + '</b>: reinícialo para aceptar peticiones.</div>') +
      '<div class="inline-controls"><label for="invoke-payload">Payload JSON</label>' +
      '<span class="signature">' + esc(endpoint.method_signature || '') + '</span></div>' +
      '<textarea class="wizard-json" id="invoke-payload" spellcheck="false" style="min-height:150px">' +
      esc(JSON.stringify(example, null, 2)) + '</textarea>' +
      '<p class="meta-line">Atajos aceptados: <code>input</code>, <code>inputs</code> o <code>arguments</code>. ' +
      'También puedes añadir <code>timeout</code> en segundos.</p>' +
      '<div class="form-actions">' +
      '<button class="btn btn-primary" type="button" id="send-inference"' + (running ? '' : ' disabled') + '>Enviar petición</button>' +
      '<button class="btn btn-secondary" type="button" id="reset-payload">Restablecer ejemplo</button>' +
      '<button class="btn btn-ghost" type="button" data-endpoint-action="copy-payload">Copiar payload</button>' +
      '<span class="status-note" id="invoke-status" role="status" aria-live="polite"></span>' +
      '</div>' +
      '<div id="invoke-result" hidden style="margin-top:14px">' +
      '<div class="code-head"><span class="code-label" id="invoke-result-title">Respuesta</span>' +
      '<button class="btn btn-ghost btn-xs" type="button" data-copy-target="#invoke-output">Copiar respuesta</button></div>' +
      '<pre class="code-highlight code-output" id="invoke-output"></pre>' +
      '<p class="meta-line" id="invoke-detail" hidden></p></div>';

    var send = byId('send-inference');
    if (send) send.addEventListener('click', sendInference);
    var reset = byId('reset-payload');
    if (reset) {
      reset.addEventListener('click', function () {
        var area = byId('invoke-payload');
        if (area) area.value = JSON.stringify(endpoint.example_payload || {}, null, 2);
      });
    }
    var copy = panel.querySelector('[data-endpoint-action="copy-payload"]');
    if (copy) {
      copy.addEventListener('click', function () {
        var area = byId('invoke-payload');
        if (area) copyText(area.value, 'Payload copiado');
      });
    }
    var copyOutput = panel.querySelector('[data-copy-target]');
    if (copyOutput) {
      copyOutput.addEventListener('click', function () {
        var target = document.querySelector(copyOutput.getAttribute('data-copy-target'));
        if (target) copyText(target.textContent, 'Respuesta copiada');
      });
    }
  }

  function sendInference() {
    var area = byId('invoke-payload');
    var status = byId('invoke-status');
    var result = byId('invoke-result');
    var output = byId('invoke-output');
    var detail = byId('invoke-detail');
    var button = byId('send-inference');
    var payload;
    try {
      payload = JSON.parse(area.value || '{}');
    } catch (error) {
      setStatus(status, 'El payload no es JSON válido: ' + error.message, 'is-error');
      return;
    }
    setStatus(status, 'Enviando petición…', 'is-busy');
    if (button) button.disabled = true;
    var started = Date.now();
    api('/api/serving/endpoints/' + encodeURIComponent(state.endpoint.id) + '/predict', {
      method: 'POST',
      body: payload
    }).then(function (data) {
      var elapsed = Date.now() - started;
      result.hidden = false;
      byId('invoke-result-title').textContent = 'Respuesta · 200 · ' + elapsed + ' ms';
      output.textContent = JSON.stringify(data, null, 2);
      detail.hidden = !data.request_id;
      detail.textContent = 'request_id: ' + (data.request_id || '—') + ' · latencia del servidor: ' + fmtMs(data.latency_ms || 0);
      setStatus(status, 'Inferencia completada', 'is-ok');
      toast('Inferencia completada en ' + fmtMs(data.latency_ms || elapsed));
      refreshEvents();
    }).catch(function (error) {
      result.hidden = false;
      byId('invoke-result-title').textContent = 'Error · ' + (error.status || '—');
      output.textContent = JSON.stringify(error.data || { error: error.message }, null, 2);
      detail.hidden = false;
      detail.textContent = 'La petición falló; el evento quedó registrado si el endpoint existe.';
      setStatus(status, error.message, 'is-error');
      refreshEvents();
    }).then(function () {
      if (button) button.disabled = false;
    });
  }

  function refreshEvents() {
    if (!state.endpoint) return;
    api('/api/serving/endpoints/' + encodeURIComponent(state.endpoint.id) + '/events?limit=100')
      .then(function (data) {
        state.events = data.events || [];
        if (state.detailTab === 'actividad') renderActivity();
      }).catch(function () { /* the next poll refreshes the view */ });
  }


  /* ------------------------------------------------------------------ */
  /* API usage documentation                                             */
  /* ------------------------------------------------------------------ */

  function renderApiPanel() {
    var panel = byId('detail-panel-api');
    if (!panel) return;
    if (!state.docs || state.docs.endpoint_id !== state.endpoint.id) {
      panel.innerHTML = '<div class="loader"><span class="spinner"></span> Generando la documentación desde el modelo…</div>';
      api('/api/serving/endpoints/' + encodeURIComponent(state.endpoint.id) + '/docs').then(function (data) {
        state.docs = data.docs;
        renderApiPanel();
      }).catch(function (error) {
        panel.innerHTML = emptyState('⚠', 'No se pudo generar la documentación', error.message);
      });
      return;
    }
    var docs = state.docs;
    var language = state.docsLanguage || 'python';
    var sample = docs.samples.filter(function (item) { return item.language === language; })[0] || docs.samples[0];
    var fields = (docs.request_schema && docs.request_schema.fields) || [];
    panel.innerHTML =
      '<div class="info-strip" style="margin-bottom:14px">' +
      '<span><strong>Deployment:</strong> <code>' + esc(docs.deployment_id || docs.endpoint_id) + '</code></span>' +
      '<span><strong>Model ID:</strong> <code>' + esc(docs.model_id || '—') + '</code></span>' +
      '<span><strong>Tipo:</strong> <code>' + esc(docs.model_type_id || docs.model.model_type_id || '—') + '</code></span>' +
      '</div>' +
      '<div class="info-strip' + (docs.auth.required ? ' is-warn' : '') + '" style="margin-bottom:14px">' +
      (docs.auth.required
        ? '<strong>Autenticación requerida.</strong> ' + esc(docs.auth.note)
        : 'Este endpoint es accesible desde la red configurada del servidor; protege la API con <code>LM_API_TOKEN</code> si lo expones fuera de localhost.') +
      '</div>' +
      '<div class="endpoint-url-bar" style="border:1px solid var(--db-line);border-radius:var(--db-r-sm);margin-bottom:14px">' +
      '<span class="endpoint-url-label">POST</span>' +
      '<span class="endpoint-url mono" title="' + esc(docs.endpoint_url) + '">' + esc(docs.endpoint_url) + '</span>' +
      '<button class="btn btn-secondary btn-sm" type="button" data-docs-copy="' + esc(docs.endpoint_url) + '">Copiar URL</button>' +
      '</div>' +
      '<div class="docs-language-bar" role="group" aria-label="Lenguaje del ejemplo">' +
      docs.samples.map(function (item) {
        return '<button class="docs-language" type="button" data-docs-language="' + esc(item.language) + '" aria-pressed="' +
          (item.language === sample.language ? 'true' : 'false') + '">' + esc(item.label) + '</button>';
      }).join('') +
      '</div>' +
      (docs.auth.required
        ? '<div class="code-head"><span class="code-label">Cabeceras de autenticación</span>' +
          '<button class="btn btn-ghost btn-xs" type="button" data-docs-copy="' +
          esc(JSON.stringify(docs.headers_example, null, 2)) + '">Copiar</button></div>' +
          '<pre class="code-highlight">' + esc(JSON.stringify(docs.headers_example, null, 2)) + '</pre>'
        : '') +
      '<div class="code-head" style="margin-top:12px"><span class="code-label">Ejemplo listo para ejecutar · ' + esc(sample.label) + '</span>' +
      '<button class="btn btn-ghost btn-xs" type="button" data-docs-copy-code="' + esc(sample.code) + '">Copiar código</button></div>' +
      '<pre class="code-highlight" id="docs-code">' + highlightCode(sample.code, sample.language) + '</pre>' +
      '<div class="chart-grid is-2" style="margin-top:16px">' +
      '<section class="panel"><div class="panel-header"><h2>Solicitud</h2>' +
      '<span class="panel-sub">' + esc(docs.method.signature || docs.method.name) + '</span></div>' +
      '<div class="panel-body" style="padding:0"><table class="schema-table"><thead><tr><th>Campo</th><th>Tipo</th><th>Obligatorio</th><th>Descripción</th></tr></thead><tbody>' +
      (fields.length
        ? fields.map(function (field) {
            return '<tr><td class="mono">' + esc(field.name) + '</td><td class="mono">' + esc(field.type || 'Any') +
              '</td><td>' + (field.required ? '<span class="schema-required">Sí</span>' : 'No') + '</td>' +
              '<td>' + esc(field.description || '—') + '<div class="mono" style="color:var(--db-faint);margin-top:2px">ej: ' +
              esc(JSON.stringify(field.example)) + '</div></td></tr>';
          }).join('')
        : '<tr><td colspan="4">El método no declara argumentos.</td></tr>') +
      '</tbody></table></div>' +
      '<div class="panel-body" style="padding-top:0"><p class="meta-line">Sobre <code>input</code>, <code>inputs</code> y <code>arguments</code>: ' +
      esc(docs.request_schema.envelope.input) + ' ' + esc(docs.request_schema.envelope.inputs) + ' ' +
      esc(docs.request_schema.envelope.arguments) + '</p></div></section>' +
      '<section class="panel"><div class="panel-header"><h2>Respuesta</h2><span class="panel-sub">JSON</span></div>' +
      '<div class="panel-body"><div class="code-head"><span class="code-label">Ejemplo de respuesta</span>' +
      '<button class="btn btn-ghost btn-xs" type="button" data-docs-copy="' + esc(JSON.stringify(docs.response_example, null, 2)) + '">Copiar</button></div>' +
      '<pre class="code-highlight">' + esc(JSON.stringify(docs.response_example, null, 2)) + '</pre>' +
      '<p class="meta-line" style="margin-top:8px">El campo <code>prediction</code> contiene la proyección JSON real del valor devuelto por ' +
      '<span class="mono">' + esc(docs.method.name) + '</span>; el tipo se informa en <code>prediction_type</code>.</p>' +
      '</div></section></div>' +
      '<div class="code-head" style="margin-top:14px"><span class="code-label">Ejemplo de payload generado</span>' +
      '<button class="btn btn-ghost btn-xs" type="button" data-docs-copy="' + esc(JSON.stringify(docs.payload_example, null, 2)) + '">Copiar payload</button></div>' +
      '<pre class="code-highlight">' + esc(JSON.stringify(docs.payload_example, null, 2)) + '</pre>';

    each(panel.querySelectorAll('[data-docs-language]'), function (button) {
      button.addEventListener('click', function () {
        state.docsLanguage = button.getAttribute('data-docs-language');
        renderApiPanel();
      });
    });
    each(panel.querySelectorAll('[data-docs-copy-code]'), function (button) {
      button.addEventListener('click', function () { copyText(button.getAttribute('data-docs-copy-code'), 'Código copiado'); });
    });
    each(panel.querySelectorAll('[data-docs-copy]'), function (button) {
      button.addEventListener('click', function () { copyText(button.getAttribute('data-docs-copy'), 'Copiado'); });
    });
  }


  function highlightCode(code, language) {
    var source = String(code || '');
    if (language === 'http') {
      return source.split('\n').map(function (line) {
        var match = line.match(/^([A-Z]+)\s+(.*)$/);
        if (match) return '<span class="tok-keyword">' + esc(match[1]) + '</span> ' + esc(match[2]);
        var header = line.match(/^([A-Za-z-]+):(.*)$/);
        if (header) return '<span class="tok-key">' + esc(header[1]) + '</span>:' + esc(header[2]);
        return esc(line);
      }).join('\n');
    }
    var patterns = {
      python: /(#[^\n]*)|("""[\s\S]*?"""|'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")|\b(import|from|as|def|return|print|with|if|else|for|in|True|False|None|await|async|raise|try|except)\b|\b(\d+(?:\.\d+)?)\b/g,
      javascript: /(\/\/[^\n]*)|('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|`(?:\\.|[^`\\])*`)|\b(const|let|var|await|async|function|return|throw|new|if|else|fetch|console|true|false|null)\b|\b(\d+(?:\.\d+)?)\b/g,
      bash: /(#[^\n]*)|('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")|\b(curl|echo|export|Bearer)\b|(--?[A-Za-z][A-Za-z-]*)/g
    };
    var regex = patterns[language] || patterns.python;
    var classes = language === 'bash'
      ? ['tok-comment', 'tok-string', 'tok-keyword', 'tok-flag']
      : ['tok-comment', 'tok-string', 'tok-keyword', 'tok-number'];
    var result = '';
    var cursor = 0;
    var match = regex.exec(source);
    while (match) {
      result += esc(source.slice(cursor, match.index));
      var groupIndex = 0;
      for (var index = 1; index <= classes.length; index += 1) {
        if (match[index] !== undefined) { groupIndex = index - 1; break; }
      }
      result += '<span class="' + classes[groupIndex] + '">' + esc(match[0]) + '</span>';
      cursor = match.index + match[0].length;
      if (!match[0].length) regex.lastIndex += 1;
      match = regex.exec(source);
    }
    return result + esc(source.slice(cursor));
  }

  /* ------------------------------------------------------------------ */
  /* Request history                                                     */
  /* ------------------------------------------------------------------ */

  function renderActivity() {
    var panel = byId('detail-panel-actividad');
    if (!panel) return;
    var events = state.events || [];
    panel.innerHTML =
      '<div class="inline-controls">' +
      '<label for="event-filter">Estado</label>' +
      '<select class="select" id="event-filter"><option value="">Todos</option>' +
      '<option value="success">Correctas</option><option value="error">Fallidas</option></select>' +
      '<button class="btn btn-secondary btn-sm" type="button" id="refresh-events">Actualizar</button>' +
      '<a class="btn btn-ghost btn-sm" href="/monitoring/' + encodeURIComponent(state.endpoint.id) + '">Abrir dashboard del endpoint</a>' +
      '</div>' +
      '<div id="event-table">' + eventTable(events) + '</div>' +
      '<div id="event-detail" hidden style="margin-top:12px"></div>';
    var filter = byId('event-filter');
    if (filter) filter.addEventListener('change', function () {
      var wanted = filter.value;
      var filtered = events.filter(function (event) { return !wanted || event.status === wanted; });
      var table = byId('event-table');
      if (table) table.innerHTML = eventTable(filtered);
    });
    var refresh = byId('refresh-events');
    if (refresh) refresh.addEventListener('click', refreshEvents);
    var body = byId('event-table');
    if (body) {
      body.addEventListener('click', function (event) {
        var row = event.target.closest('tr[data-event-id]');
        if (!row) return;
        var found = events.filter(function (item) { return item.id === row.getAttribute('data-event-id'); })[0];
        if (found) renderEventDetail(found);
      });
    }
  }

  function eventTable(events) {
    if (!events.length) {
      return emptyState('◷', 'Sin peticiones todavía', 'Cuando alguien invoque el endpoint, la latencia, el estado y los errores aparecerán aquí.');
    }
    return '<div class="table-wrap"><table class="table"><caption class="sr-only">Historial de peticiones</caption>' +
      '<thead><tr><th>Fecha</th><th>Estado</th><th class="num-head">HTTP</th><th class="num-head">Latencia</th>' +
      '<th>Método</th><th>Entrada</th><th>Error</th></tr></thead><tbody>' +
      events.map(function (event) {
        return '<tr class="event-row" data-event-id="' + esc(event.id) + '">' +
          '<td class="mono" style="font-size:10.5px">' + esc(isoDate(event.at)) + '</td>' +
          '<td><span class="event-status ' + (event.status === 'success' ? 'is-ok' : 'is-error') + '">' +
          (event.status === 'success' ? '✓ Correcta' : '✕ Fallida') + '</span></td>' +
          '<td class="cell-num">' + esc(event.http_status || '—') + '</td>' +
          '<td class="cell-num">' + fmtMs(event.latency_ms || 0) + '</td>' +
          '<td class="mono">' + esc(event.method || '—') + '</td>' +
          '<td class="mono" style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(event.input_preview || '—') + '</td>' +
          '<td class="event-error" title="' + esc(event.error_message || '') + '">' + esc(event.error_message || '') + '</td></tr>';
      }).join('') + '</tbody></table></div>';
  }

  function renderEventDetail(event) {
    var box = byId('event-detail');
    if (!box) return;
    box.hidden = false;
    box.innerHTML = '<section class="panel"><div class="panel-header"><h2>Petición ' + esc(event.id) + '</h2>' +
      '<button class="btn btn-ghost btn-xs" type="button" id="close-event-detail">Cerrar</button></div>' +
      '<div class="panel-body"><div class="review-grid">' +
      reviewCard('Fecha', isoDate(event.at)) +
      reviewCard('Estado', event.status + ' · HTTP ' + (event.http_status || '—')) +
      reviewCard('Latencia', fmtMs(event.latency_ms || 0)) +
      reviewCard('Método', event.method || '—') +
      reviewCard('Endpoint', event.endpoint_id || '—') +
      reviewCard('Modelo', event.model_id || '—') +
      '</div>' +
      '<div class="code-head" style="margin-top:12px"><span class="code-label">Entrada</span></div>' +
      '<pre class="code-highlight">' + esc(event.input_preview || '—') + '</pre>' +
      (event.output_preview ? '<div class="code-head" style="margin-top:10px"><span class="code-label">Salida</span></div>' +
        '<pre class="code-highlight">' + esc(event.output_preview) + '</pre>' : '') +
      (event.error_message ? '<div class="wizard-alert is-error" style="margin-top:10px"><strong>' + esc(event.error_type || 'Error') +
        '</strong><br>' + esc(event.error_message) + '</div>' : '') +
      (event.error_detail ? '<div class="code-head" style="margin-top:10px"><span class="code-label">Detalle del error</span></div>' +
        '<pre class="code-highlight">' + esc(event.error_detail) + '</pre>' : '') +
      '</div></section>';
    var close = byId('close-event-detail');
    if (close) close.addEventListener('click', function () { box.hidden = true; });
  }


  /* ------------------------------------------------------------------ */
  /* File browser and confirmation dialogs                               */
  /* ------------------------------------------------------------------ */

  var fileCallback = null;
  var fileSelected = '';

  function openFileBrowser(callback) {
    if (!dom.fileDialog) return;
    fileCallback = callback;
    fileSelected = state.loadPath || '';
    dom.fileDialog.hidden = false;
    window.requestAnimationFrame(function () { dom.fileDialog.classList.add('is-open'); });
    document.body.classList.add('db-body-locked');
    var start = state.loadPath && state.loadPath.indexOf('/') !== -1
      ? state.loadPath.replace(/\/[^/]*$/, '')
      : '';
    loadFiles(start);
  }

  function closeFileBrowser() {
    if (!dom.fileDialog) return;
    dom.fileDialog.classList.remove('is-open');
    window.setTimeout(function () {
      dom.fileDialog.hidden = true;
      document.body.classList.remove('db-body-locked');
    }, 160);
    fileCallback = null;
  }

  function loadFiles(path) {
    var list = byId('file-list');
    if (list) list.innerHTML = '<div class="loader"><span class="spinner"></span> Leyendo…</div>';
    api('/api/files' + (path ? '?path=' + encodeURIComponent(path) : '')).then(function (data) {
      state.filePath = data.path;
      if (byId('file-path')) byId('file-path').textContent = data.path;
      if (byId('file-relative')) byId('file-relative').textContent = data.relative || '';
      var entries = data.entries || [];
      var html = entries.map(function (entry) {
        var isDir = entry.kind === 'dir';
        return '<button class="file-entry' + (fileSelected === entry.path ? ' is-selected' : '') + '" type="button" data-file-path="' +
          esc(entry.path) + '" data-file-kind="' + (isDir ? 'dir' : 'file') + '">' +
          '<span class="file-icon">' + (isDir ? '📁' : '📄') + '</span>' +
          '<span class="file-name mono">' + esc(entry.name) + '</span>' +
          '<span class="file-size">' + (isDir ? 'carpeta' : (entry.size !== null ? fmtBytes(entry.size) : '')) + '</span></button>';
      }).join('');
      if (list) list.innerHTML = html || '<p class="empty" style="padding:18px">Directorio vacío.</p>';
      var up = byId('file-up');
      if (up) up.disabled = !data.parent;
      up.dataset.parent = data.parent || '';
      if (byId('file-selected')) {
        byId('file-selected').textContent = fileSelected || 'Directorio actual: ' + data.path;
      }
      each(list ? list.querySelectorAll('.file-entry') : [], function (button) {
        button.addEventListener('click', function () {
          var kind = button.getAttribute('data-file-kind');
          fileSelected = button.getAttribute('data-file-path');
          if (kind === 'dir') {
            loadFiles(fileSelected);
            return;
          }
          each(list.querySelectorAll('.file-entry'), function (other) { other.classList.remove('is-selected'); });
          button.classList.add('is-selected');
          if (byId('file-selected')) byId('file-selected').textContent = fileSelected;
        });
      });
    }).catch(function (error) {
      if (list) list.innerHTML = emptyState('⚠', 'No se pudo leer el directorio', error.message);
    });
  }

  function fmtBytes(value) {
    var size = Number(value || 0);
    if (size >= 1048576) return (size / 1048576).toFixed(1) + ' MB';
    if (size >= 1024) return (size / 1024).toFixed(1) + ' KB';
    return size + ' B';
  }

  var confirmCallback = null;

  function confirmAction(title, text, callback) {
    if (!dom.confirmDialog) return;
    confirmCallback = callback;
    if (byId('confirm-title')) byId('confirm-title').textContent = title;
    if (byId('confirm-text')) byId('confirm-text').innerHTML = text;
    dom.confirmDialog.hidden = false;
    window.requestAnimationFrame(function () { dom.confirmDialog.classList.add('is-open'); });
    document.body.classList.add('db-body-locked');
  }

  function closeConfirm() {
    if (!dom.confirmDialog) return;
    dom.confirmDialog.classList.remove('is-open');
    window.setTimeout(function () {
      dom.confirmDialog.hidden = true;
      document.body.classList.remove('db-body-locked');
    }, 160);
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
    var create = byId('create-endpoint');
    if (create) create.addEventListener('click', function () { openWizard(); });
    var side = byId('side-new-endpoint');
    if (side) side.addEventListener('click', function () { openWizard(); });
    var close = byId('deploy-close');
    if (close) close.addEventListener('click', closeWizard);
    var back = byId('wizard-back');
    if (back) back.addEventListener('click', function () {
      captureWizard();
      if (state.step > 1) setStep(state.step - 1);
    });
    var next = byId('wizard-next');
    if (next) next.addEventListener('click', nextStep);
    each(document.querySelectorAll('#wizard-steps .wizard-step'), function (button) {
      button.addEventListener('click', function () {
        var step = Number(button.getAttribute('data-step'));
        if (button.disabled || step > state.step) return;
        captureWizard();
        setStep(step);
      });
    });
    if (dom.deployDialog) {
      dom.deployDialog.addEventListener('click', function (event) {
        if (event.target === dom.deployDialog) closeWizard();
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
    var fileClose = byId('file-close');
    if (fileClose) fileClose.addEventListener('click', closeFileBrowser);
    var fileUp = byId('file-up');
    if (fileUp) fileUp.addEventListener('click', function () { loadFiles(fileUp.dataset.parent || ''); });
    var fileAccept = byId('file-accept');
    if (fileAccept) {
      fileAccept.addEventListener('click', function () {
        var selected = fileSelected || state.filePath;
        if (!selected) { toast('Selecciona una ruta', true); return; }
        var callback = fileCallback;
        closeFileBrowser();
        if (callback) callback(selected);
      });
    }
    if (dom.fileDialog) {
      dom.fileDialog.addEventListener('click', function (event) {
        if (event.target === dom.fileDialog) closeFileBrowser();
      });
    }
    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape') return;
      if (dom.fileDialog && !dom.fileDialog.hidden) closeFileBrowser();
      else if (dom.confirmDialog && !dom.confirmDialog.hidden) closeConfirm();
      else if (dom.deployDialog && !dom.deployDialog.hidden) closeWizard();
    });
    document.addEventListener('click', function (event) {
      var copy = event.target.closest('[data-copy-json]');
      if (copy) {
        copyText(copy.getAttribute('data-copy-json'), 'Parámetros copiados');
        return;
      }
      if (event.target.closest('#copy-method-example')) {
        var target = byId('method-example');
        if (target) copyText(target.textContent, 'Ejemplo copiado');
      }
    });
  }

  function bindFilters() {
    if (dom.search) {
      dom.search.addEventListener('input', renderEndpoints);
    }
    if (dom.statusFilter) dom.statusFilter.addEventListener('change', renderEndpoints);
    if (dom.modelFilter) dom.modelFilter.addEventListener('change', renderEndpoints);
    var refresh = byId('refresh-serving');
    if (refresh) refresh.addEventListener('click', function () {
      loadEndpoints();
      toast('Endpoints actualizados');
    });
    window.addEventListener('hashchange', function () {
      if (!state.endpoint) return;
      var hash = (window.location.hash || '').replace('#', '');
      if (['overview', 'probar', 'api', 'actividad'].indexOf(hash) !== -1) switchTab(hash);
    });
  }

  function boot() {
    dom.toast = byId('toast');
    dom.serverStatus = byId('server-status');
    dom.serverStatusText = byId('server-status-text');
    dom.sideVersion = byId('side-version');
    dom.listView = byId('serving-list-view');
    dom.detailView = byId('serving-detail-view');
    dom.detail = byId('endpoint-detail');
    dom.kpis = byId('serving-kpis');
    dom.results = byId('endpoint-results');
    dom.search = byId('endpoint-search');
    dom.statusFilter = byId('endpoint-status');
    dom.modelFilter = byId('endpoint-model');
    dom.resultBar = byId('endpoint-result-bar');
    dom.resultCount = byId('endpoint-result-count');
    dom.resultHint = byId('endpoint-result-hint');
    dom.countEndpoints = byId('count-endpoints');
    dom.crumbCurrent = byId('crumb-current');
    dom.deployDialog = byId('deploy-dialog');
    dom.confirmDialog = byId('confirm-dialog');
    dom.fileDialog = byId('file-dialog');
    dom.wizardStatus = byId('wizard-status');
    dom.wizardNext = byId('wizard-next');
    dom.wizardBack = byId('wizard-back');

    bindSidebar();
    bindTheme();
    bindEndpointTable();
    bindWizard();
    bindDialogs();
    bindFilters();
    if (dom.detail) bindDetailActions();
    loadHealth();
    route();

    var params = new URLSearchParams(window.location.search);
    var preset = params.get('model');
    if (preset) window.setTimeout(function () { openWizard(preset); }, 120);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();

