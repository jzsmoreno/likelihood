'use strict';

/*
 * Model detail page behaviour: playground, method invocation, saved-model
 * loading, tabs, section navigation and the copy/theme helpers.
 *
 * The playground, invoke and load specifications are embedded by the server as
 * JSON (<script type="application/json">), so this file never needs to know
 * anything about the model beyond those payloads.
 */
(function () {
  function byId(id) {
    return document.getElementById(id);
  }

  function each(list, callback) {
    Array.prototype.forEach.call(list || [], callback);
  }

  function closest(element, selector) {
    while (element && element.nodeType === 1) {
      if (element.matches && element.matches(selector)) return element;
      element = element.parentNode;
    }
    return null;
  }

  function esc(value) {
    var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' };
    return String(value === null || value === undefined ? '' : value)
      .replace(/[&<>"']/g, function (ch) { return map[ch]; });
  }

  function parseJsonScript(id) {
    var node = byId(id);
    if (!node) return null;
    try {
      var text = node.textContent.trim();
      return text ? JSON.parse(text) : null;
    } catch (error) {
      return null;
    }
  }

  var toastNode = byId('toast');

  function toast(message, isError) {
    if (!toastNode) return;
    toastNode.textContent = message;
    toastNode.classList.toggle('is-error', Boolean(isError));
    toastNode.classList.add('is-visible');
    window.clearTimeout(toast.timer);
    toast.timer = window.setTimeout(function () {
      toastNode.classList.remove('is-visible');
    }, 2600);
  }
  toast.timer = 0;

  function legacyCopy(value, onDone, onFail) {
    var area = document.createElement('textarea');
    area.value = value;
    area.setAttribute('readonly', 'readonly');
    area.setAttribute('aria-hidden', 'true');
    area.style.position = 'fixed';
    area.style.top = '-1000px';
    document.body.appendChild(area);
    var copied = false;
    try {
      area.select();
      copied = document.execCommand('copy');
    } catch (error) {
      copied = false;
    }
    document.body.removeChild(area);
    if (copied) { onDone(); } else { onFail(); }
  }

  function copyText(value, okMessage) {
    var text = String(value === null || value === undefined ? '' : value);
    if (!text.trim()) {
      toast('No hay nada que copiar', true);
      return;
    }
    var done = function () { toast(okMessage || 'Copiado'); };
    var failed = function () { toast('No se pudo copiar el contenido', true); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () {
        legacyCopy(text, done, failed);
      });
      return;
    }
    legacyCopy(text, done, failed);
  }

  /* One delegated listener serves every copy button on the page. */
  document.addEventListener('click', function (event) {
    if (!event.target.closest) return;
    var button = event.target.closest('[data-copy]');
    if (!button) return;
    var target = document.querySelector(button.getAttribute('data-copy'));
    if (!target) {
      toast('No se encontró el contenido a copiar', true);
      return;
    }
    copyText(target.textContent, button.getAttribute('data-toast') || 'Copiado');
  });

  /* Status line helper (playground / invoke / load). */
  function setStatus(node, message, kind) {
    if (!node) return;
    node.textContent = message || '';
    node.className = 'status-note' + (kind ? ' ' + kind : '');
  }

  /* ------------------------------------------------------------------ */
  /* Shell: sidebar, theme, section navigation                           */
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
    if (overlay) {
      overlay.hidden = !(mobile && document.body.classList.contains('sidebar-open'));
    }
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
        if (isMobileLayout()) {
          document.body.classList.toggle('sidebar-open');
        } else {
          document.body.classList.toggle('sidebar-collapsed');
        }
        syncSidebarToggle();
      });
    }
    if (overlay) overlay.addEventListener('click', closeMobileSidebar);
    each(document.querySelectorAll('#toc a'), function (link) {
      link.addEventListener('click', function () {
        if (isMobileLayout()) closeMobileSidebar();
      });
    });
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
      try {
        window.localStorage.setItem('lm-theme', next);
      } catch (error) {
        /* storage is optional */
      }
      toast(next === 'dark' ? 'Tema oscuro activado' : 'Tema claro activado');
    });
  }

  /* Hide the sections the server rendered empty (no attributes, no related). */
  function hideEmptySections() {
    var attributes = byId('atributos');
    var attributesLink = byId('toc-atributos');
    if (attributes && !attributes.querySelector('.param-card')) {
      attributes.hidden = true;
      if (attributesLink) attributesLink.hidden = true;
    }
    if (!byId('relacionados')) {
      var wrap = byId('relacionados-wrap');
      var relatedLink = byId('toc-relacionados');
      if (wrap) wrap.hidden = true;
      if (relatedLink) relatedLink.hidden = true;
    }
  }

  function bindToc() {
    var toc = byId('toc');
    if (!toc) return;
    var sourceDetails = byId('source-details');
    each(toc.querySelectorAll('a[data-section="codigo"]'), function (link) {
      link.addEventListener('click', function () {
        if (sourceDetails) sourceDetails.open = true;
      });
    });
    if (sourceDetails && window.location.hash === '#codigo') sourceDetails.open = true;

    var links = Array.prototype.slice.call(toc.querySelectorAll('a[data-section]'));
    var entries = links.map(function (link) {
      return { link: link, section: byId(link.getAttribute('data-section')) };
    }).filter(function (entry) {
      return entry.section && !entry.section.hidden && !entry.link.hidden;
    });
    if (!entries.length) return;

    var active = null;
    var scheduled = false;

    function paint() {
      scheduled = false;
      var best = entries[0];
      entries.forEach(function (entry) {
        if (entry.section.getBoundingClientRect().top <= 160) best = entry;
      });
      if (best === active) return;
      active = best;
      entries.forEach(function (entry) {
        var selected = entry === best;
        entry.link.classList.toggle('is-active', selected);
        if (selected) {
          entry.link.setAttribute('aria-current', 'true');
        } else {
          entry.link.removeAttribute('aria-current');
        }
      });
    }

    function schedule() {
      if (scheduled) return;
      scheduled = true;
      if (window.requestAnimationFrame) window.requestAnimationFrame(paint);
      else window.setTimeout(paint, 16);
    }

    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    paint();
  }

  function bindBackTop() {
    var button = byId('back-top');
    if (!button) return;
    window.addEventListener('scroll', function () {
      button.hidden = window.pageYOffset < 400;
    }, { passive: true });
    button.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  function bindPrint() {
    var details = byId('source-details');
    if (!details) return;
    window.addEventListener('beforeprint', function () { details.open = true; });
  }

  /* ------------------------------------------------------------------ */
  /* Lab tabs and shared form helpers                                    */
  /* ------------------------------------------------------------------ */

  var labTabs = Array.prototype.slice.call(document.querySelectorAll('[data-lab-tab]'));

  function selectLabTab(name, scroll) {
    var active = name === 'invoke' ? 'invoke' : 'playground';
    labTabs.forEach(function (tab) {
      var selected = tab.getAttribute('data-lab-tab') === active;
      tab.setAttribute('aria-selected', selected ? 'true' : 'false');
      tab.setAttribute('tabindex', selected ? '0' : '-1');
    });
    var panels = {
      playground: byId('panel-playground'),
      invoke: byId('panel-invoke')
    };
    Object.keys(panels).forEach(function (key) {
      if (panels[key]) panels[key].hidden = key !== active;
    });
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, '', active === 'invoke' ? '#invocar' : '#playground');
    }
    if (scroll) {
      var heading = byId('laboratorio');
      if (heading) heading.scrollIntoView({ block: 'start' });
    }
  }

  function bindLabTabs() {
    labTabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        selectLabTab(tab.getAttribute('data-lab-tab'), false);
      });
      tab.addEventListener('keydown', function (event) {
        if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
        event.preventDefault();
        var index = labTabs.indexOf(tab);
        var offset = event.key === 'ArrowRight' ? 1 : labTabs.length - 1;
        var next = labTabs[(index + offset) % labTabs.length];
        if (!next) return;
        next.focus();
        selectLabTab(next.getAttribute('data-lab-tab'), false);
      });
    });
    if (window.location.hash === '#invocar') selectLabTab('invoke', false);
  }

  /* Readable summary of the constructor parameters that need Python objects. */
  function unsupportedSummary(list) {
    return (list || []).map(function (item) {
      return "'" + item.name + "' (" + item.type + ': ' + item.reason + ')';
    }).join(', ');
  }

  function fieldControl(field, index, prefix) {
    var id = prefix + '-' + index;
    var kind = field.kind;
    var value = field.default;
    if ((value === null || value === undefined) && field.required) value = field.sample;
    var placeholder = field.required ? 'obligatorio' : 'opcional';
    if (kind === 'bool') {
      return '<label class="field-check" for="' + id + '">' +
        '<input type="checkbox" id="' + id + '" data-kind="bool"' +
        (value === true ? ' checked' : '') + '> activar</label>';
    }
    if (kind === 'int' || kind === 'float') {
      return '<input type="number" id="' + id + '" data-kind="' + kind + '" step="' +
        (kind === 'int' ? '1' : 'any') + '" value="' +
        esc(value === null || value === undefined ? '' : String(value)) +
        '" placeholder="' + placeholder + '">';
    }
    if (kind === 'raw') {
      return '<textarea id="' + id + '" data-kind="raw" spellcheck="false"' +
        ' placeholder="JSON: [1, 2, 3]">' +
        esc(value === null || value === undefined ? '' : JSON.stringify(value)) +
        '</textarea>';
    }
    return '<input type="text" id="' + id + '" data-kind="' + kind + '" value="' +
      esc(value === null || value === undefined ? '' : String(value)) +
      '" placeholder="' + placeholder + '">';
  }

  function renderFields(container, fields, prefix, emptyMessage) {
    if (!container) return;
    if (!fields.length) {
      container.innerHTML = '<p class="empty">' + esc(emptyMessage) + '</p>';
      return;
    }
    container.innerHTML = fields.map(function (field, index) {
      return '<div class="form-field">' +
        '<label for="' + prefix + '-' + index + '">' + esc(field.name) +
        (field.required ? ' *' : '') + '</label>' +
        '<span class="field-type">' + esc(field.type) + '</span>' +
        fieldControl(field, index, prefix) + '</div>';
    }).join('');
  }

  function readControl(control) {
    var kind = control.getAttribute('data-kind');
    if (kind === 'bool') return control.checked;
    var text = control.value;
    if (kind === 'raw') {
      if (!text.trim()) return null;
      return JSON.parse(text);
    }
    if (kind === 'int' || kind === 'float') {
      if (!text.trim()) return null;
      var number = Number(text);
      if (!isFinite(number)) throw new Error('se esperaba un número');
      return kind === 'int' ? Math.trunc(number) : number;
    }
    return text === '' ? null : text;
  }

  function collectFields(spec, prefix) {
    var values = {};
    var missing = [];
    var invalid = [];
    ((spec && spec.fields) || []).forEach(function (field, index) {
      var control = byId(prefix + '-' + index);
      if (!control) return;
      var value;
      try {
        value = readControl(control);
      } catch (error) {
        invalid.push(field.name + ': ' + error.message);
        return;
      }
      if (value === null) {
        if (field.required) missing.push(field.name);
        return;
      }
      values[field.name] = value;
    });
    return { values: values, missing: missing, invalid: invalid };
  }

  /* ------------------------------------------------------------------ */
  /* Payloads and helpers shared by the three tools                      */
  /* ------------------------------------------------------------------ */

  var playgroundData = parseJsonScript('playground-data');
  var invokeData = parseJsonScript('invoke-data');
  var loadData = parseJsonScript('load-data');

  /* Keep the raw control state so re-rendering a form never discards input. */
  function snapshotFields(fields, prefix) {
    var snapshot = {};
    (fields || []).forEach(function (field, index) {
      var control = byId(prefix + '-' + index);
      if (!control) return;
      snapshot[field.name] = control.getAttribute('data-kind') === 'bool'
        ? { bool: true, checked: control.checked }
        : { bool: false, value: control.value };
    });
    return snapshot;
  }

  function restoreFields(fields, prefix, snapshot) {
    if (!snapshot) return;
    (fields || []).forEach(function (field, index) {
      var saved = snapshot[field.name];
      if (!saved) return;
      var control = byId(prefix + '-' + index);
      if (!control) return;
      if (saved.bool) {
        control.checked = saved.checked;
      } else {
        control.value = saved.value;
      }
    });
  }

  /* Stable signature of the constructor: key order must not force a rebuild. */
  function canonicalValue(value) {
    if (Object.prototype.toString.call(value) === '[object Array]') {
      return value.map(canonicalValue);
    }
    if (value && typeof value === 'object') {
      var sorted = {};
      Object.keys(value).sort().forEach(function (key) {
        sorted[key] = canonicalValue(value[key]);
      });
      return sorted;
    }
    return value;
  }

  function constructorSignature(values) {
    return JSON.stringify(canonicalValue(values || {}));
  }

  function pyLiteral(value) {
    if (value === null || value === undefined) return 'None';
    if (typeof value === 'boolean') return value ? 'True' : 'False';
    if (typeof value === 'number') return String(value);
    if (typeof value === 'string') return JSON.stringify(value);
    if (Object.prototype.toString.call(value) === '[object Array]') {
      return '[' + value.map(pyLiteral).join(', ') + ']';
    }
    if (typeof value === 'object') {
      return '{' + Object.keys(value).map(function (key) {
        return JSON.stringify(key) + ': ' + pyLiteral(value[key]);
      }).join(', ') + '}';
    }
    return JSON.stringify(value);
  }

  function postJson(url, payload) {
    return window.fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json().catch(function () { return null; }).then(function (data) {
        return { ok: response.ok, status: response.status, data: data || {} };
      });
    });
  }

  function describeInvokeError(data) {
    var message = data.error || 'no se pudo completar la operación';
    if (data.unknown && data.unknown.length) {
      message += ' (argumentos desconocidos: ' + data.unknown.join(', ') + ')';
    }
    if (data.missing && data.missing.length) {
      message += ' (faltan: ' + data.missing.join(', ') + ')';
    }
    return message;
  }

  /* ------------------------------------------------------------------ */
  /* Playground                                                          */
  /* ------------------------------------------------------------------ */

  var playgroundDom = {
    form: byId('playground-form'),
    fields: byId('playground-fields'),
    run: byId('run-playground'),
    reset: byId('reset-playground'),
    status: byId('playground-status'),
    result: byId('playground-result'),
    title: byId('playground-result-title'),
    output: byId('playground-output'),
    detail: byId('playground-detail'),
    code: byId('playground-code')
  };

  function collectParameters() {
    return collectFields(playgroundData, 'pg');
  }

  function updateSnippet() {
    if (!playgroundData || !playgroundDom.code) return;
    var collected = collectParameters();
    var callArguments = Object.keys(collected.values).map(function (name) {
      return name + '=' + pyLiteral(collected.values[name]);
    });
    playgroundDom.code.textContent = (playgroundData.import_statement || '') +
      '\n\nmodel = ' + playgroundData.model_name + '(' + callArguments.join(', ') + ')';
  }

  function renderPlayground() {
    if (!playgroundData || !playgroundDom.fields) return;
    if (playgroundData.buildable === false) {
      playgroundDom.fields.innerHTML =
        '<p class="empty">No se puede construir desde el catálogo: ' +
        esc(unsupportedSummary(playgroundData.unsupported)) +
        '. Usa el ejemplo de Python.</p>';
      if (playgroundDom.run) playgroundDom.run.disabled = true;
      if (playgroundDom.reset) playgroundDom.reset.disabled = true;
      updateSnippet();
      return;
    }
    renderFields(
      playgroundDom.fields,
      playgroundData.fields || [],
      'pg',
      'Este modelo no declara parámetros configurables.'
    );
    updateSnippet();
  }

  function showPlaygroundResult(text, detail, ok) {
    if (!playgroundDom.result) return;
    playgroundDom.result.hidden = false;
    if (playgroundDom.title) {
      playgroundDom.title.textContent = ok ? 'Resultado de la ejecución' : 'No se pudo ejecutar';
    }
    playgroundDom.output.textContent = text;
    playgroundDom.detail.hidden = !detail;
    playgroundDom.detail.textContent = detail || '';
  }

  function runPlayground(event) {
    event.preventDefault();
    if (!playgroundData || playgroundData.buildable === false) return;
    var collected = collectParameters();
    if (collected.invalid.length) {
      setStatus(playgroundDom.status, 'Revisa: ' + collected.invalid.join('; '), 'is-error');
      return;
    }
    if (collected.missing.length) {
      setStatus(
        playgroundDom.status,
        'Faltan parámetros obligatorios: ' + collected.missing.join(', '),
        'is-error'
      );
      return;
    }
    playgroundDom.run.disabled = true;
    setStatus(playgroundDom.status, 'Ejecutando…', 'is-busy');
    window.fetch(playgroundData.endpoint || '/api/playground', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: playgroundData.model_id,
        parameters: collected.values
      })
    }).then(function (response) {
      return response.json().catch(function () { return null; }).then(function (data) {
        return { ok: response.ok, data: data || {} };
      });
    }).then(function (response) {
      var data = response.data;
      if (!response.ok || data.success === false) {
        setStatus(
          playgroundDom.status,
          'Error: ' + (data.error || 'no se pudo ejecutar el modelo'),
          'is-error'
        );
        showPlaygroundResult(data.error || 'Error desconocido', data.detail || '', false);
        return;
      }
      var lines = [
        'tipo   : ' + (data.type || '—'),
        'módulo : ' + (data.module || '—'),
        'repr   : ' + (data.repr || '—')
      ];
      if (data.attributes && data.attributes.length) {
        lines.push('atributos: ' + data.attributes.join(', '));
      }
      if (data.stdout) lines.push('', 'salida:', data.stdout);
      var elapsed = typeof data.elapsed === 'number' ? ' en ' + data.elapsed + ' s' : '';
      setStatus(playgroundDom.status, 'Modelo construido' + elapsed, 'is-ok');
      showPlaygroundResult(lines.join('\n'), data.stderr || '', true);
    }).catch(function (error) {
      setStatus(playgroundDom.status, 'Error de red: ' + error.message, 'is-error');
      showPlaygroundResult(String(error.message || error), '', false);
    }).then(function () {
      playgroundDom.run.disabled = false;
    });
  }

  function bindPlayground() {
    if (playgroundData && playgroundDom.form) {
      renderPlayground();
      playgroundDom.form.addEventListener('submit', runPlayground);
      /* Every edit is mirrored into the invoke constructor and refreshes the
         dummy inputs, so the examples never describe stale values. */
      var onPlaygroundEdit = function () {
        mirrorConstructorFields('pg', 'ic');
        updateSnippet();
        updateInvokeSnippet();
        refreshInvokeSamples('pg');
      };
      playgroundDom.form.addEventListener('input', onPlaygroundEdit);
      playgroundDom.form.addEventListener('change', onPlaygroundEdit);
      if (playgroundDom.reset) {
        playgroundDom.reset.addEventListener('click', function () {
          renderPlayground();
          playgroundDom.result.hidden = true;
          setStatus(playgroundDom.status, 'Valores restablecidos', '');
        });
      }
      return;
    }
    if (playgroundDom.fields) {
      playgroundDom.fields.innerHTML =
        '<p class="empty">El playground no está disponible: falta la especificación del modelo.</p>';
    }
    if (playgroundDom.run) playgroundDom.run.disabled = true;
  }

  /* ------------------------------------------------------------------ */
  /* Load a saved model                                                  */
  /* ------------------------------------------------------------------ */

  var loadDom = {
    panel: byId('load-panel'),
    select: byId('load-loader'),
    path: byId('load-path'),
    browse: byId('load-browse'),
    useSaved: byId('load-use-saved'),
    savedRow: byId('load-saved-row'),
    saved: byId('load-saved'),
    reset: byId('load-reset'),
    resetConfirm: byId('load-reset-confirm'),
    resetYes: byId('load-reset-yes'),
    resetCancel: byId('load-reset-cancel'),
    verify: byId('load-verify'),
    use: byId('load-session'),
    hint: byId('load-hint'),
    status: byId('load-status'),
    result: byId('load-result'),
    title: byId('load-result-title'),
    output: byId('load-output'),
    detail: byId('load-detail')
  };

  function setLoadStatus(message, kind) {
    setStatus(loadDom.status, message, kind);
  }

  function showLoadResult(text, detail, ok) {
    if (!loadDom.result) return;
    loadDom.result.hidden = false;
    if (loadDom.title) loadDom.title.textContent = ok ? 'Modelo cargado' : 'No se pudo cargar';
    loadDom.output.textContent = text;
    loadDom.detail.hidden = !detail;
    loadDom.detail.textContent = detail || '';
  }

  function currentLoader() {
    if (!loadData || !loadDom.select) return null;
    var loaders = loadData.loaders || [];
    for (var index = 0; index < loaders.length; index += 1) {
      if (loaders[index].name === loadDom.select.value) return loaders[index];
    }
    return loaders.length ? loaders[0] : null;
  }

  function loadConstructorValues() {
    var fromInvoke = collectFields(playgroundData, 'ic');
    if (fromInvoke.invalid.length || fromInvoke.missing.length) return collectParameters();
    return fromInvoke;
  }

  function loadPayload(loader) {
    var values = {};
    if (loader && loader.requires_instance) {
      var constructor = loadConstructorValues();
      if (constructor.invalid.length) {
        setLoadStatus('Revisa el constructor: ' + constructor.invalid.join('; '), 'is-error');
        return null;
      }
      if (constructor.missing.length) {
        setLoadStatus(
          'Este loader necesita el constructor: faltan ' + constructor.missing.join(', '),
          'is-error'
        );
        return null;
      }
      values = constructor.values;
    }
    var path = loadDom.path ? loadDom.path.value.trim() : '';
    if (!path) {
      setLoadStatus('Escribe la ruta del modelo guardado', 'is-error');
      return null;
    }
    return { loader: loader ? loader.name : '', path: path, parameters: values };
  }

  function verifyLoadedModel() {
    if (!loadData || !loadData.loadable || loadData.enabled === false) return;
    var loader = currentLoader();
    var payload = loadPayload(loader);
    if (!payload) return;
    if (loadDom.verify) loadDom.verify.disabled = true;
    if (loadDom.use) loadDom.use.disabled = true;
    setLoadStatus('Cargando el modelo…', 'is-busy');
    postJson(loadData.endpoint, payload)
      .then(function (response) {
        var data = response.data;
        if (!response.ok || data.success === false) {
          setLoadStatus('Error: ' + describeInvokeError(data), 'is-error');
          showLoadResult(data.error || 'Error desconocido', data.detail || '', false);
          return;
        }
        var lines = [
          'tipo   : ' + (data.type || '—'),
          'loader : ' + (data.loader || payload.loader || '—'),
          'ruta   : ' + (data.path || payload.path || '—')
        ];
        if (data.attributes && data.attributes.length) {
          lines.push('atributos: ' + data.attributes.join(', '));
        }
        var elapsed = typeof data.elapsed === 'number' ? ' en ' + data.elapsed + ' s' : '';
        setLoadStatus('Modelo verificado' + elapsed + ' · ejemplos sincronizados', 'is-ok');
        showLoadResult(lines.join('\n'), data.stderr || '', true);
        rememberLoadedModel({
          path: data.path || payload.path,
          loader: data.loader || payload.loader,
          parameters: payload.parameters
        });
        synchronizeWithParameters(payload.parameters, true);
      })
      .catch(function (error) {
        setLoadStatus('Error de red: ' + error.message, 'is-error');
        showLoadResult(String(error.message || error), '', false);
      })
      .then(function () {
        if (loadDom.verify) loadDom.verify.disabled = false;
        if (loadDom.use) loadDom.use.disabled = false;
      });
  }

  function loadAsSession() {
    if (!loadData || !loadData.loadable || loadData.enabled === false || !invokeData) return;
    var loader = currentLoader();
    var payload = loadPayload(loader);
    if (!payload) return;
    if (loadDom.verify) loadDom.verify.disabled = true;
    if (loadDom.use) loadDom.use.disabled = true;
    setLoadStatus('Abriendo una instancia con el modelo guardado…', 'is-busy');
    var start = invokeSession.id ? closeSession(false) : Promise.resolve();
    start.then(function () {
      return postJson(invokeData.sessions_url, payload);
    }).then(function (response) {
      var data = response.data;
      if (!response.ok || data.success === false) {
        throw new Error(describeInvokeError(data));
      }
      invokeSession.id = data.session_id;
      invokeSession.calls = 0;
      invokeSession.signature = null;
      invokeSession.loaded = true;
      invokeSession.loader = payload.loader;
      invokeSession.path = data.load ? data.load.path : payload.path;
      if (invokeDom.run) invokeDom.run.disabled = false;
      if (invokeDom.sample) invokeDom.sample.disabled = false;
      if (invokeDom.select) invokeDom.select.disabled = false;
      if (invokeDom.useSession) invokeDom.useSession.checked = true;
      renderSessionInfo();
      renderInvokeConstructor();
      renderInvokeMethod();
      setLoadStatus('Instancia cargada y lista para invocar', 'is-ok');
      setInvokeStatus(
        'Instancia cargada desde ' + (data.load ? data.load.path : payload.path),
        'is-ok'
      );
      rememberLoadedModel({
        path: invokeSession.path,
        loader: invokeSession.loader,
        parameters: payload.parameters
      });
      synchronizeWithParameters(payload.parameters, true);
      selectLabTab('invoke', true);
    }).catch(function (error) {
      setLoadStatus('Error: ' + error.message, 'is-error');
    }).then(function () {
      if (loadDom.verify) loadDom.verify.disabled = false;
      if (loadDom.use) loadDom.use.disabled = false;
    });
  }

  function renderLoad() {
    if (!loadDom.panel || !loadData) return;
    var loaders = loadData.loaders || [];
    if (loadData.enabled === false || !loadData.loadable || !loaders.length) {
      loadDom.panel.hidden = true;
      return;
    }
    loadDom.panel.hidden = false;
    if (loadDom.select) {
      loadDom.select.innerHTML = loaders.map(function (loader) {
        var suffix = loader.requires_instance ? ' · instancia' : '';
        return '<option value="' + esc(loader.name) + '">' +
          esc(loader.name + '()' + suffix) + '</option>';
      }).join('');
    }
    var browser = loadData.browser || {};
    if (loadDom.hint) {
      var policy = loadData.restrict_to_root
        ? 'Solo se permiten rutas dentro de ' + loadData.root + '.'
        : 'Rutas relativas a ' + loadData.root + '.';
      var browseHint = browser.enabled
        ? ' Usa <b>Examinar…</b> para elegirla con el navegador de archivos' +
          ' (limitado a <code>' + esc(browser.root || loadData.root) + '</code>).'
        : '';
      loadDom.hint.innerHTML = esc(policy) + browseHint +
        ' Se aceptan archivos <code>.pkl</code> y directorios de checkpoints.' +
        ' Carga únicamente archivos de confianza.';
    }
    if (loadDom.browse) {
      loadDom.browse.hidden = !browser.enabled;
      loadDom.browse.addEventListener('click', openFileDialog);
    }
    if (loadDom.path && !loadDom.path.value) {
      try {
        loadDom.path.value = window.sessionStorage.getItem('lm-load-path') || '';
      } catch (error) {
        loadDom.path.value = '';
      }
    }
    if (loadDom.path) loadDom.path.addEventListener('change', rememberLoadPath);
    if (loadDom.select) loadDom.select.addEventListener('change', rememberLoadPath);
    if (loadDom.useSaved) {
      loadDom.useSaved.addEventListener('change', function () {
        context.usePath = Boolean(loadDom.useSaved.checked);
        persistContext();
        renderSavedState();
        if (!context.usePath && invokeSession.loaded) closeSession(false);
      });
    }
    if (loadDom.reset) loadDom.reset.addEventListener('click', openResetConfirm);
    if (loadDom.resetYes) loadDom.resetYes.addEventListener('click', resetModelConfiguration);
    if (loadDom.resetCancel) loadDom.resetCancel.addEventListener('click', cancelResetConfirm);
    if (loadDom.verify) loadDom.verify.addEventListener('click', verifyLoadedModel);
    if (loadDom.use) {
      var sessionsAvailable = !invokeData || invokeData.sessions_enabled !== false;
      loadDom.use.disabled = !sessionsAvailable;
      loadDom.use.title = sessionsAvailable
        ? 'Carga el modelo en una instancia persistente'
        : 'Las sesiones están deshabilitadas en este servidor';
      loadDom.use.addEventListener('click', loadAsSession);
    }
    restoreSavedContext();
  }

  /* ------------------------------------------------------------------ */
  /* Saved model context (path, loader and constructor values)           */
  /* ------------------------------------------------------------------ */

  var CONTEXT_PREFIX = 'lm-model-context:';
  var context = { path: '', loader: '', usePath: false, parameters: null };

  function modelId() {
    if (playgroundData && playgroundData.model_id) return playgroundData.model_id;
    if (invokeData && invokeData.model_id) return invokeData.model_id;
    return 'model';
  }

  function contextKey() {
    return CONTEXT_PREFIX + modelId();
  }

  function normalizeContext(raw) {
    if (!raw || typeof raw !== 'object') return null;
    var path = String(raw.path || '').trim();
    if (!path) return null;
    return {
      path: path,
      loader: String(raw.loader || ''),
      usePath: raw.usePath !== false,
      parameters: raw.parameters && typeof raw.parameters === 'object' ? raw.parameters : null
    };
  }

  function readStoredContext() {
    try {
      var raw = window.localStorage.getItem(contextKey());
      if (raw) return normalizeContext(JSON.parse(raw));
    } catch (error) {
      /* storage or JSON unavailable: fall through to the legacy key */
    }
    try {
      var legacy = window.sessionStorage.getItem('lm-load-path') || '';
      if (legacy) return normalizeContext({ path: legacy, usePath: true });
    } catch (error) {
      /* storage is optional */
    }
    return null;
  }

  function persistContext() {
    try {
      if (!context.path) {
        window.localStorage.removeItem(contextKey());
        return;
      }
      window.localStorage.setItem(
        contextKey(),
        JSON.stringify({
          path: context.path,
          loader: context.loader || '',
          usePath: Boolean(context.usePath),
          parameters: context.parameters || null,
          updatedAt: new Date().toISOString()
        })
      );
    } catch (error) {
      /* storage is optional */
    }
  }

  function forgetContext() {
    context = { path: '', loader: '', usePath: false, parameters: null };
    try {
      window.localStorage.removeItem(contextKey());
      window.sessionStorage.removeItem('lm-load-path');
    } catch (error) {
      /* storage is optional */
    }
  }

  function selectedLoaderName() {
    return loadDom.select ? loadDom.select.value || '' : '';
  }

  /* Constructor values of the catalogue: what the model uses by default. */
  function defaultParameters() {
    var values = {};
    ((playgroundData && playgroundData.fields) || []).forEach(function (field) {
      var value = field.default;
      if ((value === null || value === undefined) && field.required) value = field.sample;
      if (value !== null && value !== undefined) values[field.name] = value;
    });
    return values;
  }

  /* The saved model is only used while the user keeps it enabled. */
  function activeSource() {
    if (!context.path || !context.usePath) return null;
    return { path: context.path, loader: context.loader || selectedLoaderName() };
  }

  function rememberLoadPath() {
    if (!loadDom.path) return;
    var value = loadDom.path.value.trim();
    context.path = value;
    context.loader = selectedLoaderName() || context.loader;
    context.usePath = Boolean(value);
    if (!value) context.parameters = null;
    /* Editing the path invalidates the live instance created from the old one. */
    if (invokeSession.loaded && (invokeSession.path || '') !== value) closeSession(false);
    persistContext();
    renderSavedState();
  }

  function renderSavedState() {
    var hasPath = Boolean(context.path);
    if (loadDom.savedRow) loadDom.savedRow.hidden = !hasPath;
    if (loadDom.useSaved) loadDom.useSaved.checked = hasPath && context.usePath;
    if (loadDom.saved) {
      loadDom.saved.hidden = !hasPath;
      loadDom.saved.innerHTML = hasPath
        ? '<b>Modelo guardado:</b> <code>' + esc(context.path) + '</code>' +
          (context.usePath ? '' : ' — no se usará en las peticiones')
        : '';
    }
    if (loadDom.reset) loadDom.reset.disabled = false;
  }

  function restoreSavedContext() {
    var stored = readStoredContext();
    if (stored) context = stored;
    if (context.path && loadDom.path) loadDom.path.value = context.path;
    if (context.loader && loadDom.select && loadDom.select.options) {
      var known = Array.prototype.some.call(loadDom.select.options, function (option) {
        return option.value === context.loader;
      });
      if (known) loadDom.select.value = context.loader;
    }
    if (context.path && loadDom.panel) loadDom.panel.open = true;
    renderSavedState();
    if (context.path) {
      setLoadStatus('Modelo guardado restaurado: ' + context.path, 'is-ok');
    }
  }

  /* Persist the artifact that was just verified/loaded. */
  function rememberLoadedModel(payload) {
    var path = String(payload.path || (loadDom.path ? loadDom.path.value : '') || '').trim();
    if (!path) return;
    context.path = path;
    context.loader = String(payload.loader || selectedLoaderName() || '');
    context.usePath = true;
    context.parameters =
      payload.parameters && Object.keys(payload.parameters).length ? payload.parameters : null;
    if (loadDom.path) loadDom.path.value = path;
    if (loadDom.useSaved) loadDom.useSaved.checked = true;
    persistContext();
    renderSavedState();
  }

  /* ------------------------------------------------------------------ */
  /* Examples and playground stay in sync with the loaded model          */
  /* ------------------------------------------------------------------ */

  function restoreFieldValues(fields, prefix, values) {
    (fields || []).forEach(function (field, index) {
      if (!Object.prototype.hasOwnProperty.call(values || {}, field.name)) return;
      var control = byId(prefix + '-' + index);
      if (control) fillField(control, values[field.name]);
    });
  }

  /* Rebuild both constructor forms with ``parameters`` (or their defaults). */
  function applyParameters(parameters) {
    var values = parameters || {};
    renderPlayground();
    restoreFieldValues((playgroundData && playgroundData.fields) || [], 'pg', values);
    updateSnippet();
    renderInvokeConstructor(true);
    restoreFieldValues((playgroundData && playgroundData.fields) || [], 'ic', values);
    updateInvokeSnippet();
  }

  function applySamples(method, samples) {
    (method.fields || []).forEach(function (field, index) {
      if (!Object.prototype.hasOwnProperty.call(samples, field.name)) return;
      var control = byId('iv-' + index);
      if (control) fillField(control, samples[field.name]);
    });
  }

  /* Ask the server for dummy inputs shaped by ``parameters``.  Only the newest
     response is applied, so a slow request never overwrites fresher examples. */
  var sampleRequestSeq = 0;

  function requestSamples(parameters, done) {
    var method = currentInvokeMethod();
    if (!invokeData || !method) {
      if (done) done(false);
      return;
    }
    var endpoint = invokeData.sample_endpoint || '/api/models/' + invokeData.model_id + '/sample';
    sampleRequestSeq += 1;
    var seq = sampleRequestSeq;
    postJson(endpoint, { method: method.name, parameters: parameters || {} })
      .then(function (response) {
        if (seq !== sampleRequestSeq) {
          if (done) done(false);
          return;
        }
        if (!response.ok || response.data.success === false) {
          if (done) done(false);
          return;
        }
        applySamples(method, response.data.arguments || {});
        updateInvokeSnippet();
        if (done) done(true);
      })
      .catch(function () {
        if (done) done(false);
      });
  }

  /* Examples are only valid for the constructor values they were built with, so
     editing them regenerates the dummy inputs (never the other way around). */
  var SAMPLE_REFRESH_DELAY = 300;
  var sampleRefreshTimer = 0;

  function cancelSampleRefresh() {
    if (!sampleRefreshTimer) return;
    window.clearTimeout(sampleRefreshTimer);
    sampleRefreshTimer = 0;
  }

  /* Values of one constructor form, or ``null`` while the user is still typing. */
  function constructorValues(prefix) {
    var collected = collectFields(playgroundData, prefix);
    if (collected.invalid.length || collected.missing.length) return null;
    return collected.values;
  }

  function refreshInvokeSamples(prefix) {
    cancelSampleRefresh();
    if (!invokeData || !currentInvokeMethod()) return;
    var values = constructorValues(prefix || 'ic');
    if (values === null) return;
    sampleRefreshTimer = window.setTimeout(function () {
      sampleRefreshTimer = 0;
      requestSamples(values, null);
    }, SAMPLE_REFRESH_DELAY);
  }

  /* Both constructor forms describe the same model: editing one mirrors the other
     so the examples, the snippet and the request always agree. */
  function copyControl(from, to) {
    if (!from || !to) return;
    if (from.getAttribute('data-kind') === 'bool') {
      to.checked = from.checked;
      return;
    }
    to.value = from.value;
  }

  function mirrorConstructorFields(source, target) {
    if (!playgroundData) return;
    (playgroundData.fields || []).forEach(function (field, index) {
      copyControl(byId(source + '-' + index), byId(target + '-' + index));
    });
  }

  /* Refresh the playground, the invoke form and the example inputs. */
  function synchronizeWithParameters(parameters, silent) {
    var values = parameters && Object.keys(parameters).length ? parameters : defaultParameters();
    cancelSampleRefresh();
    applyParameters(values);
    if (!invokeData) return;
    renderInvokeMethod();
    requestSamples(values, function (ok) {
      if (!ok || silent) return;
      setInvokeStatus('Ejemplos sincronizados con el modelo cargado', 'is-ok');
    });
  }

  function openResetConfirm() {
    if (loadDom.resetConfirm) loadDom.resetConfirm.hidden = false;
  }

  function cancelResetConfirm() {
    if (loadDom.resetConfirm) loadDom.resetConfirm.hidden = true;
  }

  /* Forget the saved model and put every control back to its catalogue default. */
  function resetModelConfiguration() {
    cancelSampleRefresh();
    forgetContext();
    if (loadDom.path) loadDom.path.value = '';
    if (loadDom.select && loadData) {
      var fallback =
        loadData.default_loader ||
        (loadData.loaders && loadData.loaders.length ? loadData.loaders[0].name : '');
      if (fallback) loadDom.select.value = fallback;
    }
    if (invokeSession.id) closeSession(false);
    applyParameters(defaultParameters());
    if (invokeData) {
      renderInvokeMethod();
      requestSamples(defaultParameters(), null);
    }
    if (playgroundDom.result) playgroundDom.result.hidden = true;
    if (invokeDom.result) invokeDom.result.hidden = true;
    if (loadDom.result) loadDom.result.hidden = true;
    renderSavedState();
    cancelResetConfirm();
    setLoadStatus('Modelo guardado restablecido', 'is-ok');
    setStatus(playgroundDom.status, '', '');
    setInvokeStatus('Valores por defecto restablecidos', 'is-ok');
    toast('Configuración restablecida');
  }

  /* ------------------------------------------------------------------ */
  /* Saved-model file browser                                            */
  /* ------------------------------------------------------------------ */

  var fileDom = {
    backdrop: byId('file-dialog'),
    list: byId('file-dialog-list'),
    path: byId('file-dialog-path'),
    status: byId('file-dialog-status'),
    close: byId('file-dialog-close'),
    cancel: byId('file-dialog-cancel'),
    up: byId('file-dialog-up'),
    root: byId('file-dialog-root'),
    hidden: byId('file-dialog-hidden'),
    selectDir: byId('file-dialog-select-dir')
  };

  var fileState = { current: null, trigger: null, busy: false };

  function fileEndpoint() {
    var browser = (loadData && loadData.browser) || {};
    return browser.endpoint || '/api/files';
  }

  function formatBytes(value) {
    if (typeof value !== 'number') return '';
    if (value < 1024) return value + ' B';
    var units = ['KB', 'MB', 'GB', 'TB'];
    var size = value / 1024;
    var index = 0;
    while (size >= 1024 && index < units.length - 1) {
      size /= 1024;
      index += 1;
    }
    return size.toFixed(size >= 10 ? 0 : 1) + ' ' + units[index];
  }

  function formatDate(value) {
    return value ? String(value).replace('T', ' ').slice(0, 16) : '';
  }

  function fileRow(entry, selected) {
    var icon = entry.kind === 'dir' ? '📁' : '📄';
    var meta = entry.kind === 'dir' ? 'carpeta' : formatBytes(entry.size);
    var badge = entry.artifact ? '<span class="badge badge-green">modelo</span>' : '';
    return '<button class="file-row" type="button" role="option"' +
      ' data-file-path="' + esc(entry.path) + '" data-file-kind="' + esc(entry.kind) + '"' +
      ' aria-selected="' + (selected ? 'true' : 'false') + '" title="' + esc(entry.path) + '">' +
      '<span class="file-row-icon" aria-hidden="true">' + icon + '</span>' +
      '<span class="file-row-name">' + esc(entry.name) + '</span>' +
      '<span class="file-row-meta">' + esc(meta + ' ' + formatDate(entry.modified)) + '</span>' +
      badge + '</button>';
  }

  function renderListing(payload) {
    fileState.current = payload;
    if (fileDom.path) {
      fileDom.path.textContent = payload.path;
      fileDom.path.title = payload.path;
    }
    if (fileDom.up) fileDom.up.disabled = !payload.parent;
    if (fileDom.selectDir) fileDom.selectDir.disabled = !payload.path;
    var rows = (payload.entries || []).map(function (entry) {
      return fileRow(entry, entry.name === payload.selected);
    }).join('');
    if (!rows) {
      rows = '<div class="empty-state"><span class="empty-icon" aria-hidden="true">⌕</span>' +
        '<h3>Sin archivos</h3><p>Esta carpeta está vacía o solo contiene archivos ocultos.</p></div>';
    }
    if (payload.truncated) {
      rows += '<p class="status-note">Se muestran los primeros ' + esc(payload.limit) +
        ' elementos de la carpeta.</p>';
    }
    fileDom.list.innerHTML = rows;
    setStatus(fileDom.status, '', '');
  }

  function loadListing(path) {
    if (fileState.busy) return;
    fileState.busy = true;
    setStatus(fileDom.status, 'Cargando…', 'is-busy');
    var params = [];
    if (path) params.push('path=' + encodeURIComponent(path));
    if (fileDom.hidden && fileDom.hidden.checked) params.push('hidden=1');
    window.fetch(fileEndpoint() + (params.length ? '?' + params.join('&') : ''))
      .then(function (response) {
        return response.json().catch(function () { return null; }).then(function (data) {
          return { ok: response.ok, data: data || {} };
        });
      })
      .then(function (response) {
        if (!response.ok || response.data.success === false) {
          fileDom.list.innerHTML = '';
          setStatus(fileDom.status, 'Error: ' + describeInvokeError(response.data), 'is-error');
          return;
        }
        renderListing(response.data);
      })
      .catch(function (error) {
        setStatus(fileDom.status, 'Error de red: ' + error.message, 'is-error');
      })
      .then(function () {
        fileState.busy = false;
      });
  }

  function openFileDialog() {
    if (!fileDom.backdrop || !loadData) return;
    fileState.trigger = loadDom.browse || null;
    fileDom.backdrop.hidden = false;
    document.body.classList.add('db-body-locked');
    loadListing(loadDom.path ? loadDom.path.value.trim() : '');
    fileDom.list.focus();
  }

  function closeFileDialog() {
    if (!fileDom.backdrop || fileDom.backdrop.hidden) return;
    fileDom.backdrop.hidden = true;
    document.body.classList.remove('db-body-locked');
    if (fileState.trigger && document.contains(fileState.trigger)) fileState.trigger.focus();
    fileState.trigger = null;
  }

  function choosePath(path) {
    if (!path || !loadDom.path) return;
    loadDom.path.value = path;
    rememberLoadPath();
    closeFileDialog();
    setLoadStatus('Ruta seleccionada: ' + path, 'is-ok');
    toast('Ruta seleccionada');
  }

  function onFileRowClick(event) {
    var row = closest(event.target, '[data-file-path]');
    if (!row) return;
    var path = row.getAttribute('data-file-path');
    if (row.getAttribute('data-file-kind') === 'dir') {
      loadListing(path);
      return;
    }
    choosePath(path);
  }

  function trapFileDialogFocus(event) {
    var focusable = fileDom.backdrop.querySelectorAll(
      'button:not([disabled]), input, [tabindex]:not([tabindex="-1"])'
    );
    if (!focusable.length) return;
    var first = focusable[0];
    var last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function bindFileDialog() {
    if (!fileDom.backdrop) return;
    fileDom.list.addEventListener('click', onFileRowClick);
    if (fileDom.close) fileDom.close.addEventListener('click', closeFileDialog);
    if (fileDom.cancel) fileDom.cancel.addEventListener('click', closeFileDialog);
    if (fileDom.up) {
      fileDom.up.addEventListener('click', function () {
        if (fileState.current && fileState.current.parent) loadListing(fileState.current.parent);
      });
    }
    if (fileDom.root) {
      fileDom.root.addEventListener('click', function () {
        loadListing('');
      });
    }
    if (fileDom.hidden) {
      fileDom.hidden.addEventListener('change', function () {
        loadListing(fileState.current ? fileState.current.path : '');
      });
    }
    if (fileDom.selectDir) {
      fileDom.selectDir.addEventListener('click', function () {
        if (fileState.current && fileState.current.path) choosePath(fileState.current.path);
      });
    }
    fileDom.backdrop.addEventListener('mousedown', function (event) {
      if (event.target === fileDom.backdrop) closeFileDialog();
    });
    fileDom.backdrop.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' || event.key === 'Esc') {
        event.preventDefault();
        closeFileDialog();
        return;
      }
      if (event.key === 'Tab') trapFileDialogFocus(event);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Method invocation                                                   */
  /* ------------------------------------------------------------------ */

  var invokeDom = {
    panel: byId('panel-invoke'),
    select: byId('invoke-method'),
    signature: byId('invoke-signature'),
    fields: byId('invoke-fields'),
    constructorFields: byId('invoke-constructor-fields'),
    constructorNote: byId('invoke-constructor-note'),
    useSession: byId('invoke-use-session'),
    sessionInfo: byId('invoke-session-info'),
    closeSession: byId('invoke-close-session'),
    run: byId('run-invoke'),
    sample: byId('sample-invoke'),
    reset: byId('reset-invoke'),
    status: byId('invoke-status'),
    result: byId('invoke-result'),
    title: byId('invoke-result-title'),
    output: byId('invoke-output'),
    detail: byId('invoke-detail'),
    code: byId('invoke-code')
  };

  var invokeSession = { id: null, calls: 0, signature: null, loaded: false, loader: null, path: null };

  function setInvokeStatus(message, kind) {
    setStatus(invokeDom.status, message, kind);
  }

  function showInvokeResult(text, detail, ok) {
    if (!invokeDom.result) return;
    invokeDom.result.hidden = false;
    if (invokeDom.title) {
      invokeDom.title.textContent = ok ? 'Resultado de la llamada' : 'No se pudo invocar';
    }
    invokeDom.output.textContent = text;
    invokeDom.detail.hidden = !detail;
    invokeDom.detail.textContent = detail || '';
  }

  function currentInvokeMethod() {
    if (!invokeData || !invokeDom.select) return null;
    var methods = invokeData.methods || [];
    for (var index = 0; index < methods.length; index += 1) {
      if (methods[index].name === invokeDom.select.value) return methods[index];
    }
    return methods.length ? methods[0] : null;
  }

  function renderSessionInfo() {
    if (invokeDom.sessionInfo) {
      invokeDom.sessionInfo.textContent = invokeSession.id
        ? 'sesión ' + invokeSession.id + ' · ' + invokeSession.calls +
          (invokeSession.calls === 1 ? ' llamada' : ' llamadas') +
          (invokeSession.loaded ? ' · cargada' : '')
        : 'sin sesión';
    }
    if (invokeDom.closeSession) invokeDom.closeSession.hidden = !invokeSession.id;
  }

  function renderInvokeMethod() {
    if (!invokeData) return;
    var method = currentInvokeMethod();
    renderFields(
      invokeDom.fields,
      (method && method.fields) || [],
      'iv',
      'Este método no recibe argumentos.'
    );
    if (invokeDom.signature) invokeDom.signature.textContent = method ? method.signature : '';
    updateInvokeSnippet();
  }

  function renderInvokeConstructor(forceDefaults) {
    if (playgroundData && playgroundData.buildable === false) {
      if (invokeSession.loaded) {
        if (invokeDom.constructorFields) {
          invokeDom.constructorFields.innerHTML =
            '<p class="empty">Instancia cargada desde un archivo; el constructor no se usa.</p>';
        }
        if (invokeDom.constructorNote) {
          invokeDom.constructorNote.textContent =
            'Métodos disponibles sobre la instancia cargada.';
        }
        return;
      }
      if (invokeDom.constructorFields) {
        invokeDom.constructorFields.innerHTML =
          '<p class="empty">No se puede construir desde el catálogo: ' +
          esc(unsupportedSummary(playgroundData.unsupported)) +
          '. Usa el ejemplo de Python.</p>';
      }
      if (invokeDom.constructorNote) {
        invokeDom.constructorNote.textContent =
          'Este modelo necesita objetos Python que la API no puede recibir como JSON.';
      }
      return;
    }
    var fields = (playgroundData && playgroundData.fields) || [];
    /* Opening or closing a session must not reset the form: it has to keep
       showing the values of the live instance so the signature is accurate. */
    var previous = forceDefaults ? null : snapshotFields(fields, 'ic');
    renderFields(
      invokeDom.constructorFields,
      fields,
      'ic',
      'El constructor no declara parámetros: el modelo se crea sin argumentos.'
    );
    restoreFields(fields, 'ic', previous);
    if (invokeDom.constructorNote) {
      invokeDom.constructorNote.textContent = invokeSession.loaded
        ? 'Instancia cargada desde un archivo: los parámetros del constructor no se aplican.'
        : (invokeSession.id
          ? 'La instancia se reconstruye automáticamente si cambias estos valores.'
          : 'La primera llamada construye la clase con estos valores.');
    }
  }

  function updateInvokeSnippet() {
    if (!invokeData || !invokeDom.code) return;
    var method = currentInvokeMethod();
    var constructor = collectFields(playgroundData, 'ic');
    var callArguments = method ? collectFields(method, 'iv') : { values: {} };
    var constructorArguments = Object.keys(constructor.values).map(function (name) {
      return name + '=' + pyLiteral(constructor.values[name]);
    });
    var argumentsList = Object.keys(callArguments.values).map(function (name) {
      return name + '=' + pyLiteral(callArguments.values[name]);
    });
    var source = playgroundData || {};
    var importStatement = invokeData.import_statement || source.import_statement || '';
    var modelName = invokeData.model_name || source.model_name || invokeData.model_id;
    var lines = [importStatement, ''];
    if (invokeSession.loaded) {
      lines.push(
        'model = ' + modelName + '.' + (invokeSession.loader || 'load') +
        '(' + pyLiteral(invokeSession.path || '') + ')'
      );
    } else {
      lines.push('model = ' + modelName + '(' + constructorArguments.join(', ') + ')');
    }
    if (method) {
      lines.push('result = model.' + method.name + '(' + argumentsList.join(', ') + ')');
      lines.push('print(result)');
    }
    invokeDom.code.textContent = lines.join('\n');
  }

  function renderInvoke() {
    if (!invokeData) {
      if (invokeDom.fields) {
        invokeDom.fields.innerHTML =
          '<p class="empty">La invocación no está disponible: falta la especificación del modelo.</p>';
      }
      if (invokeDom.run) invokeDom.run.disabled = true;
      if (invokeDom.sample) invokeDom.sample.disabled = true;
      return;
    }
    if (invokeDom.select) {
      invokeDom.select.innerHTML = (invokeData.methods || []).map(function (method) {
        return '<option value="' + esc(method.name) + '">' + esc(method.name) + '</option>';
      }).join('');
      invokeDom.select.disabled = !(invokeData.methods || []).length;
    }
    if (invokeDom.useSession) {
      invokeDom.useSession.disabled = !invokeData.sessions_enabled;
      if (!invokeData.sessions_enabled) invokeDom.useSession.checked = false;
    }
    if (invokeDom.sample) {
      invokeDom.sample.disabled = !(invokeData.methods || []).length;
    }
    renderInvokeConstructor();
    renderSessionInfo();
    if (invokeData.enabled === false) {
      if (invokeDom.fields) {
        invokeDom.fields.innerHTML =
          '<p class="empty">La invocación de métodos está deshabilitada en este servidor ' +
          '(LM_INVOKE=0).</p>';
      }
      if (invokeDom.run) invokeDom.run.disabled = true;
      if (invokeDom.sample) invokeDom.sample.disabled = true;
      setInvokeStatus('Deshabilitada en el servidor', 'is-error');
      return;
    }
    if (!(invokeData.methods || []).length) {
      if (invokeDom.fields) {
        invokeDom.fields.innerHTML =
          '<p class="empty">Este modelo no expone métodos invocables con la política actual.</p>';
      }
      if (invokeDom.run) invokeDom.run.disabled = true;
      setInvokeStatus('Sin métodos invocables', '');
      return;
    }
    if (invokeData.buildable === false && !invokeSession.loaded && !activeSource()) {
      if (invokeDom.fields) {
        invokeDom.fields.innerHTML =
          '<p class="empty">No se puede construir desde el catálogo: ' +
          esc(unsupportedSummary(invokeData.unsupported)) +
          '. Usa el ejemplo de Python.</p>';
      }
      if (invokeDom.run) invokeDom.run.disabled = true;
      if (invokeDom.sample) invokeDom.sample.disabled = true;
      if (invokeDom.select) invokeDom.select.disabled = true;
      setInvokeStatus('Requiere objetos Python', 'is-error');
      return;
    }
    if (invokeDom.run) invokeDom.run.disabled = false;
    renderInvokeMethod();
  }

  function openSession(parameters) {
    return postJson(invokeData.sessions_url, { parameters: parameters }).then(function (response) {
      if (!response.ok || response.data.success === false) {
        throw new Error(describeInvokeError(response.data));
      }
      invokeSession.id = response.data.session_id;
      invokeSession.calls = 0;
      invokeSession.signature = constructorSignature(parameters);
      invokeSession.loaded = false;
      invokeSession.loader = null;
      invokeSession.path = null;
      renderSessionInfo();
      renderInvokeConstructor();
      return invokeSession.id;
    });
  }

  /* Open a session from a saved artifact (the persisted model path). */
  function openSessionFromPath(source, parameters) {
    var payload = { loader: source.loader || '', path: source.path };
    if (parameters && Object.keys(parameters).length) payload.parameters = parameters;
    return postJson(invokeData.sessions_url, payload).then(function (response) {
      if (!response.ok || response.data.success === false) {
        throw new Error(describeInvokeError(response.data));
      }
      invokeSession.id = response.data.session_id;
      invokeSession.calls = 0;
      invokeSession.signature = null;
      invokeSession.loaded = true;
      invokeSession.loader = payload.loader;
      invokeSession.path = response.data.load ? response.data.load.path : payload.path;
      renderSessionInfo();
      renderInvokeConstructor();
      return invokeSession.id;
    });
  }

  function closeSession(notify) {
    var sessionId = invokeSession.id;
    invokeSession.id = null;
    invokeSession.calls = 0;
    invokeSession.signature = null;
    invokeSession.loaded = false;
    invokeSession.loader = null;
    invokeSession.path = null;
    renderSessionInfo();
    renderInvokeConstructor();
    if (!sessionId) return Promise.resolve();
    var template = invokeData.session_url_template || '/api/sessions/{session_id}';
    return window.fetch(template.replace('{session_id}', sessionId), {
      method: 'DELETE'
    }).then(function () {
      if (notify) setInvokeStatus('Sesión cerrada', 'is-ok');
    }).catch(function () {
      /* the session expires on its own on the server */
    });
  }

  /* Fill one form control with a JSON value. */
  function fillField(control, value) {
    var kind = control.getAttribute('data-kind');
    if (kind === 'bool') {
      control.checked = value === true;
      return;
    }
    if (kind === 'raw') {
      control.value = JSON.stringify(value);
      return;
    }
    control.value = value === null || value === undefined ? '' : String(value);
  }

  function validatedConstructor() {
    var constructor = collectFields(playgroundData, 'ic');
    if (constructor.invalid.length) {
      setInvokeStatus('Revisa el constructor: ' + constructor.invalid.join('; '), 'is-error');
      return null;
    }
    if (constructor.missing.length) {
      setInvokeStatus(
        'Faltan parámetros del constructor: ' + constructor.missing.join(', '),
        'is-error'
      );
      return null;
    }
    return constructor.values;
  }

  /* Ask the server for sample arguments coherent with the chosen constructor. */
  function fillSampleArguments() {
    var method = currentInvokeMethod();
    var loaded = invokeSession.id && invokeSession.loaded;
    if (!invokeData || (invokeData.buildable === false && !loaded && !activeSource()) || !method) {
      if (invokeData && invokeData.buildable === false && !loaded && !activeSource()) {
        setInvokeStatus('Requiere objetos Python', 'is-error');
      } else {
        setInvokeStatus('No hay ningún método que invocar', 'is-error');
      }
      return;
    }
    var source = activeSource();
    var loader = currentLoader();
    var parameters;
    if (loaded || (source && loader && !loader.requires_instance)) {
      parameters = {};
    } else {
      parameters = validatedConstructor();
      if (!parameters) return;
    }
    var endpoint = invokeData.sample_endpoint || '/api/models/' + invokeData.model_id + '/sample';
    if (invokeDom.sample) invokeDom.sample.disabled = true;
    setInvokeStatus('Generando datos de ejemplo…', 'is-busy');
    postJson(endpoint, { method: method.name, parameters: parameters })
      .then(function (response) {
        var data = response.data;
        if (!response.ok || data.success === false) {
          setInvokeStatus('Error: ' + describeInvokeError(data), 'is-error');
          return;
        }
        var samples = data.arguments || {};
        applySamples(method, samples);
        updateInvokeSnippet();
        setInvokeStatus('Datos de ejemplo generados', 'is-ok');
      })
      .catch(function (error) {
        setInvokeStatus('Error de red: ' + error.message, 'is-error');
      })
      .then(function () {
        if (invokeDom.sample) invokeDom.sample.disabled = false;
      });
  }

  function runInvoke() {
    if (!invokeData) return;
    var loaded = invokeSession.id && invokeSession.loaded;
    var source = activeSource();
    if (invokeData.buildable === false && !loaded && !source) return;
    var method = currentInvokeMethod();
    if (!method) {
      setInvokeStatus('No hay ningún método que invocar', 'is-error');
      return;
    }
    var argumentsValues = collectFields(method, 'iv');
    if (argumentsValues.invalid.length) {
      setInvokeStatus('Revisa: ' + argumentsValues.invalid.join('; '), 'is-error');
      return;
    }
    if (argumentsValues.missing.length) {
      setInvokeStatus('Faltan argumentos: ' + argumentsValues.missing.join(', '), 'is-error');
      return;
    }
    /* A saved model loaded from disk does not need the constructor form unless
       its loader rebuilds the instance from the recorded values. */
    var useSession = invokeDom.useSession && invokeDom.useSession.checked;
    var loader = currentLoader();
    var needsConstructor = !source || !loader || loader.requires_instance;
    var constructorValues = { values: {} };
    if (needsConstructor && (!loaded || !useSession)) {
      constructorValues = collectFields(playgroundData, 'ic');
      if (constructorValues.invalid.length) {
        setInvokeStatus('Revisa el constructor: ' + constructorValues.invalid.join('; '), 'is-error');
        return;
      }
      if (constructorValues.missing.length) {
        setInvokeStatus(
          'Faltan parámetros del constructor: ' + constructorValues.missing.join(', '),
          'is-error'
        );
        return;
      }
    }
    var signature = constructorSignature(constructorValues.values);
    if (invokeDom.run) invokeDom.run.disabled = true;
    setInvokeStatus('Invocando…', 'is-busy');
    var request;
    if (useSession) {
      var reuse = invokeSession.id &&
        (source ? invokeSession.loaded : !invokeSession.loaded && invokeSession.signature === signature);
      var sessionPromise = reuse
        ? Promise.resolve(invokeSession.id)
        : closeSession(false).then(function () {
          return source
            ? openSessionFromPath(source, constructorValues.values)
            : openSession(constructorValues.values);
        });
      request = sessionPromise.then(function (sessionId) {
        var template = invokeData.session_invoke_url_template ||
          '/api/sessions/{session_id}/invoke';
        var payload = { method: method.name, arguments: argumentsValues.values };
        if (!invokeSession.loaded) payload.parameters = constructorValues.values;
        return postJson(template.replace('{session_id}', sessionId), payload);
      });
    } else {
      var directPayload = { method: method.name, arguments: argumentsValues.values };
      if (source) {
        directPayload.parameters = needsConstructor ? constructorValues.values : {};
        directPayload.path = source.path;
        directPayload.loader = source.loader;
      } else {
        directPayload.parameters = constructorValues.values;
      }
      request = postJson(invokeData.endpoint, directPayload);
    }
    request.then(function (response) {
      var data = response.data;
      if (!response.ok || data.success === false) {
        if (response.status === 404 && useSession && invokeSession.id) {
          invokeSession.id = null;
          invokeSession.calls = 0;
          renderSessionInfo();
        }
        setInvokeStatus('Error: ' + describeInvokeError(data), 'is-error');
        showInvokeResult(data.error || 'Error desconocido', data.detail || '', false);
        return;
      }
      if (useSession) {
        invokeSession.id = data.session_id || invokeSession.id;
        invokeSession.signature = signature;
        invokeSession.calls = typeof data.calls === 'number'
          ? data.calls
          : invokeSession.calls + 1;
        renderSessionInfo();
        renderInvokeConstructor();
      }
      var lines = [
        'método : ' + (data.method || method.name),
        'tipo   : ' + (data.result_type || '—'),
        'repr   : ' + (data.result_repr || '—')
      ];
      if (data.result !== undefined && data.result !== null) {
        lines.push('', 'resultado:', JSON.stringify(data.result, null, 2));
      }
      if (data.stdout) lines.push('', 'salida:', data.stdout);
      var elapsed = typeof data.elapsed === 'number' ? ' en ' + data.elapsed + ' s' : '';
      setInvokeStatus(
        'Llamada completada' + elapsed + (data.rebuilt ? ' (instancia reconstruida)' : ''),
        'is-ok'
      );
      showInvokeResult(lines.join('\n'), data.stderr || '', true);
    }).catch(function (error) {
      setInvokeStatus('Error: ' + error.message, 'is-error');
      showInvokeResult(String(error.message || error), '', false);
    }).then(function () {
      if (invokeDom.run) invokeDom.run.disabled = false;
    });
  }

  /* ------------------------------------------------------------------ */
  /* Bootstrap                                                           */
  /* ------------------------------------------------------------------ */

  function bindInvoke() {
    renderInvoke();
    if (invokeDom.select) {
      invokeDom.select.addEventListener('change', function () {
        renderInvokeMethod();
        refreshInvokeSamples('ic');
      });
    }
    if (invokeDom.run) invokeDom.run.addEventListener('click', runInvoke);
    if (invokeDom.sample) invokeDom.sample.addEventListener('click', fillSampleArguments);
    if (invokeDom.constructorFields) {
      /* Editing the constructor here follows the Playground and rebuilds the
         examples with the new values. */
      var onConstructorEdit = function () {
        mirrorConstructorFields('ic', 'pg');
        updateSnippet();
        updateInvokeSnippet();
        refreshInvokeSamples('ic');
      };
      invokeDom.constructorFields.addEventListener('input', onConstructorEdit);
      invokeDom.constructorFields.addEventListener('change', onConstructorEdit);
    }
    if (invokeDom.reset) {
      invokeDom.reset.addEventListener('click', function () {
        renderInvokeMethod();
        renderInvokeConstructor(true);
        if (invokeDom.result) invokeDom.result.hidden = true;
        setInvokeStatus('Argumentos restablecidos', '');
        refreshInvokeSamples('ic');
      });
    }
    if (invokeDom.useSession) {
      invokeDom.useSession.addEventListener('change', function () {
        if (!invokeDom.useSession.checked && invokeSession.id) closeSession(false);
      });
    }
    if (invokeDom.closeSession) {
      invokeDom.closeSession.addEventListener('click', function () {
        closeSession(true);
      });
    }
  }

  function init() {
    hideEmptySections();
    bindSidebar();
    bindTheme();
    bindToc();
    bindBackTop();
    bindPrint();
    bindLabTabs();
    bindPlayground();
    bindFileDialog();
    renderLoad();
    bindInvoke();
    /* A restored saved model defines the defaults shown for this page. */
    if (context.path) {
      synchronizeWithParameters(context.parameters || defaultParameters(), true);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();











