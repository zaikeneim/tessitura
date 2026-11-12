"""Finestra principale: registrazione, elaborazione, revisione e export."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import paths
from ..hardware import MODELS, Hardware, detect, estimate_text
from ..audio.capture import Recorder
from ..audio.devices import DeviceInfo, get_devices
from ..models import storage
from ..models.session import Session
from ..models.project import Project, open_or_create, load_project, is_project
from ..processing.audio_profile import AudioProfile
from ..processing.pipeline import PipelineOptions
from .audio_player import AudioPlayer
from .first_run_dialog import FirstRunDialog
from .model_settings import ModelSettingsDialog
from .update_dialog import UpdateDialog
from ..update_manager import ActiveOperationState, UpdateManager
from .project_dialog import ProjectDetailsDialog, describe_project
from .project_actions import (
    combined_glossary,
    apply_project_names,
    confirm_speakers,
    sync_project_learning,
    tag_session,
    store_session,
    summary_for_memory,
    unconfirm_speaker,
)
from .conversation_view import ConversationView
from .speaker_panel import SpeakerPanel
from .speaker_timeline_view import SpeakerTimelineView
from .transcript_controller import TranscriptController
from .transcript_view import TranscriptView
from .workers import (
    BatchProcessWorker,
    CombineAudioWorker,
    EncodeWorker,
    ProcessWorker,
    UpdateCheckWorker,
    StopRecordWorker,
)
from .recording_mode_dialog import ask_recording_mode
from .theme import icon
from .stage_view import StageView

# Lingue offerte nel menu. None = rilevamento automatico di Whisper.
LANGUAGES: list[tuple[str, str | None]] = [
    ("Automatica", None),
    ("Italiano", "it"),
    ("English", "en"),
    ("Português", "pt"),
    ("Español", "es"),
    ("Français", "fr"),
    ("Deutsch", "de"),
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tessitura — trascrizione & diarizzazione")
        self.resize(1320, 860)
        self.setMinimumSize(1000, 700)

        self.session: Session | None = None
        self.project: Project | None = None
        self.recorder = Recorder()
        self.last_mic: Path | None = None
        self.last_system: Path | None = None
        self.last_offsets: list[float] = []
        self.last_dual = False
        self.session_path: Path | None = None
        self._worker: ProcessWorker | None = None
        self._batch_worker: BatchProcessWorker | None = None
        self._encoder: EncodeWorker | None = None
        self._updater: UpdateCheckWorker | None = None
        self._stopper: StopRecordWorker | None = None
        self._combiner: CombineAudioWorker | None = None
        self._stopping = False
        self._busy = False
        self._encoding = False
        self._dirty = False
        self._showing_raw = False
        self._pipeline_strategy = "standard"
        self._headphones_confirmed = True
        self._pipeline_strategy_configured = False
        self._last_recording_config: dict | None = None
        self.controller = TranscriptController(self)
        self._previous_speaker_sidebar_visible = True
        self.controller.activeViewChanged.connect(self._on_active_view_changed)

        self._elapsed = 0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._populate_devices()
        self._set_busy(False)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("workspace")
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 12, 20, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        mark = QLabel()
        mark.setPixmap(QIcon(str(paths.app_root() / "assets" / "tessitura.ico")).pixmap(44, 44))
        header.addWidget(mark)
        brand = QLabel("Tessitura")
        brand.setObjectName("brand")
        header.addWidget(brand)
        subtitle = QLabel("Dalle conversazioni alle parole.")
        subtitle.setObjectName("muted")
        header.addWidget(subtitle)
        header.addStretch()
        badge = QLabel("Elaborazione locale")
        badge.setObjectName("badge")
        header.addWidget(badge)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(16)
        sidebar = QWidget()
        sidebar.setLayout(self._build_controls())
        sidebar.setFixedWidth(320)
        body.addWidget(sidebar)
        review = QVBoxLayout()
        title_row = QHBoxLayout()
        self.session_title = QLabel("Il tuo spazio di ascolto")
        self.session_title.setObjectName("sectionTitle")
        self.session_title.setWordWrap(True)
        title_row.addWidget(self.session_title, 1)
        self.save_text_btn.setIcon(icon("save"))
        title_row.addWidget(self.save_text_btn)
        review.addLayout(title_row)
        self.review_hint = QLabel("Registra una conversazione o importa un file audio per iniziare.")
        self.review_hint.setObjectName("muted")
        self.review_hint.setWordWrap(True)
        review.addWidget(self.review_hint)

        self.review_commands = QHBoxLayout()
        self.review_commands.setSpacing(8)

        # Barra di commutazione viste a 3 vie
        self.view_btn_group = QButtonGroup(self)
        self.view_btn_group.setExclusive(True)

        self.conv_view_btn = QToolButton()
        self.conv_view_btn.setText("Conversazione")
        self.conv_view_btn.setCheckable(True)
        self.conv_view_btn.setChecked(True)
        self.conv_view_btn.clicked.connect(lambda: self.controller.set_active_view("conversation"))
        self.conv_view_btn.setToolTip("Vista continua a fumetti conversazionali con badge per le sovrapposizioni")

        self.timeline_view_btn = QToolButton()
        self.timeline_view_btn.setText("Timeline")
        self.timeline_view_btn.setCheckable(True)
        self.timeline_view_btn.clicked.connect(lambda: self.controller.set_active_view("timeline"))
        self.timeline_view_btn.setToolTip("Vista a colonne parallele su asse temporale continuo")

        self.table_view_btn = QToolButton()
        self.table_view_btn.setText("Tabellare")
        self.table_view_btn.setCheckable(True)
        self.table_view_btn.clicked.connect(lambda: self.controller.set_active_view("table"))
        self.table_view_btn.setToolTip("Vista classica riga per riga")

        self.view_btn_group.addButton(self.conv_view_btn)
        self.view_btn_group.addButton(self.timeline_view_btn)
        self.view_btn_group.addButton(self.table_view_btn)

        self.review_commands.addWidget(self.conv_view_btn)
        self.review_commands.addWidget(self.timeline_view_btn)
        self.review_commands.addWidget(self.table_view_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #2a394d;")
        self.review_commands.addWidget(sep)
        review.addLayout(self.review_commands)

        splitter = QSplitter(Qt.Horizontal)
        self.transcript_stack = QStackedWidget()
        self.conversation_view = ConversationView(self.controller)
        self.timeline_view = SpeakerTimelineView(self.controller)
        self.transcript = TranscriptView()
        self.transcript.setAccessibleName("Trascrizione: tempo, voce e testo modificabile")

        self.transcript_stack.addWidget(self.conversation_view)
        self.transcript_stack.addWidget(self.timeline_view)
        self.transcript_stack.addWidget(self.transcript)
        splitter.addWidget(self.transcript_stack)

        right = QWidget()
        self.speaker_sidebar = right
        right.setMinimumWidth(160)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("VOCI · Assegna un nome"))
        self.speakers = SpeakerPanel()
        right_layout.addWidget(self.speakers)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([620, 220])
        splitter.setChildrenCollapsible(False)
        review.addWidget(splitter, 1)

        self.player = AudioPlayer()
        review.addWidget(self.player)
        review.addWidget(self.progress)
        self.stage_view = StageView()
        self.stage_toggle = QToolButton()
        self.stage_toggle.setText("Fasi di elaborazione")
        self.stage_toggle.setToolTip("Mostra o nascondi fasi, tempi e avvisi dell'ultima elaborazione")
        self.stage_toggle.setCheckable(True)
        self.stage_toggle.toggled.connect(self.stage_view.setVisible)
        self.stage_toggle.hide()
        review.addWidget(self.stage_toggle)
        review.addWidget(self.stage_view)
        review.addWidget(self.status_label)
        body.addLayout(review, 1)
        root.addLayout(body, 1)

        self.setCentralWidget(central)
        self._build_menu()

        # Share QAction instances: toolbar and menus always have the same busy state.
        for action, label in (
            (self.group_segments_action, "Raggruppa voci"),
            (self.raw_transcript_action, "Trascrizione grezza"),
        ):
            button = QToolButton()
            action.setIconText(label)
            button.setDefaultAction(action)
            button.setText(label)
            button.setAccessibleName(label)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.review_commands.addWidget(button)
        self.review_commands.addStretch()
        self.voices_toggle = QToolButton()
        self.voices_toggle.setText("Voci")
        self.voices_toggle.setCheckable(True)
        self.voices_toggle.setChecked(True)
        self.voices_toggle.setToolTip("Mostra o nascondi il pannello dei nomi delle voci")
        self.voices_toggle.setAccessibleName("Mostra pannello voci")
        self.voices_toggle.toggled.connect(right.setVisible)
        self.review_commands.addWidget(self.voices_toggle)

        # wiring
        self.controller.seekRequested.connect(self.player.seek_seconds)
        self.transcript.seekRequested.connect(self.player.seek_seconds)
        self.player.positionChanged.connect(self._on_player_position_changed)
        self.speakers.namesChanged.connect(self._on_speaker_names_changed)
        self.speakers.namesChanged.connect(self.transcript.refresh_speakers)
        self.speakers.namesChanged.connect(self._mark_dirty)
        self.speakers.speakerConfirmed.connect(self._on_speaker_name_confirmed)
        self.speakers.exportSpeakerRequested.connect(self._export_single_speaker)
        self.controller.utteranceEdited.connect(self._mark_dirty)
        self.controller.speakerChanged.connect(self._mark_dirty)
        self.controller.changed.connect(self._on_timeline_changed)
        self.transcript.textEdited.connect(self._refresh_canonical_views)
        self.transcript.speakerChanged.connect(self._refresh_canonical_views)
        self.timeline_view.regionPlaybackRequested.connect(self.player.play_region)
        self.timeline_view.reprocessRequested.connect(self._reprocess_region)
        self.timeline_view.audioVariantRequested.connect(self.player.select_variant)
        self.timeline_view.identityReviewRequested.connect(self._review_global_identities)
        self.player.sourceChanged.connect(self._sync_audio_source)
        self.transcript.textEdited.connect(self._mark_dirty)
        self.transcript.speakerChanged.connect(self._on_speaker_reassigned)

    def _build_controls(self) -> QVBoxLayout:
        row = QVBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.control_tabs = QTabWidget()
        self.control_tabs.setObjectName("workflowTabs")
        self.control_tabs.setAccessibleName("Strumenti: audio, trascrizione e progetto")
        self.control_tabs.setUsesScrollButtons(False)
        row.addWidget(self.control_tabs, 1)

        def page(title):
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(12)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(content)
            self.control_tabs.addTab(scroll, title)
            return layout

        audio_page = page("&Audio")
        processing_page = page("&Trascrizione")
        project_page = page("&Progetto")
        intro = QLabel("Importa un file o registra una conversazione. Prima di elaborare, scegli il profilo nella scheda Trascrizione.")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        audio_page.addWidget(intro)

        # --- progetto ---
        proj_box = QGroupBox("Progetto")
        proj_layout = QVBoxLayout(proj_box)

        self.project_label = QLabel("Nessun progetto aperto")
        self.project_label.setObjectName("sectionTitle")
        self.project_label.setWordWrap(True)
        proj_layout.addWidget(self.project_label)

        self.project_status = QLabel("Apri una cartella per collegare audio, memoria e glossario.")
        self.project_status.setObjectName("muted")
        self.project_status.setWordWrap(True)
        proj_layout.addWidget(self.project_status)

        proj_btn_row = QHBoxLayout()
        self.open_project_btn = QPushButton("Apri cartella…")
        self.open_project_btn.setIcon(icon("import"))
        self.open_project_btn.clicked.connect(self._open_project)
        self.open_project_btn.setToolTip("Apre o crea un progetto nella cartella selezionata")
        proj_btn_row.addWidget(self.open_project_btn)

        self.project_details_btn = QPushButton("Dettagli…")
        self.project_details_btn.clicked.connect(self._show_project_details)
        self.project_details_btn.setEnabled(False)
        self.project_details_btn.setToolTip("Modifica note, glossario, voci e visualizza la memoria del modello")
        proj_btn_row.addWidget(self.project_details_btn)
        proj_layout.addLayout(proj_btn_row)

        self.batch_process_btn = QPushButton("Elabora coda audio")
        self.batch_process_btn.setIcon(icon("spark", "#102c29"))
        self.batch_process_btn.clicked.connect(self._start_batch_processing)
        self.batch_process_btn.setEnabled(False)
        self.batch_process_btn.setToolTip("Trascrive e diarizza in sequenza tutti gli audio non ancora elaborati nel progetto")
        proj_layout.addWidget(self.batch_process_btn)

        project_page.addWidget(proj_box)
        project_page.addStretch()

        # --- registrazione ---
        rec_box = QGroupBox("Registrazione")
        rec_layout = QVBoxLayout(rec_box)

        self.recording_mode_combo = QComboBox()
        self.recording_mode_combo.setAccessibleName("Modalità di registrazione")
        self.recording_mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.recording_mode_combo.setMinimumContentsLength(12)
        self.recording_mode_combo.addItem("Ambientale — una o più persone nella stanza", "ambient")
        self.recording_mode_combo.addItem("Cuffie con microfono — locale e partecipanti remoti", "headset")
        self.recording_mode_combo.addItem("Microfono e altoparlanti — con possibile eco", "speakers")
        self.recording_mode_combo.currentIndexChanged.connect(self._recording_mode_changed)
        rec_layout.addWidget(self.recording_mode_combo)

        self.mic_check = QCheckBox("Microfono")
        self.mic_check.setChecked(True)
        self.mic_combo = QComboBox()
        self.mic_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mic_combo.setMinimumContentsLength(12)
        self.mic_combo.setAccessibleName("Dispositivo microfono")
        self.mic_combo.currentIndexChanged.connect(self._capture_device_changed)
        mic_row = QVBoxLayout()
        mic_row.addWidget(self.mic_check)
        mic_row.addWidget(self.mic_combo, 1)
        rec_layout.addLayout(mic_row)

        self.sys_check = QCheckBox("Acquisisci uscita (anche cuffie)")
        self.sys_check.setChecked(True)
        self.sys_combo = QComboBox()
        self.sys_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.sys_combo.setMinimumContentsLength(12)
        self.sys_combo.setAccessibleName("Dispositivo di uscita per loopback")
        self.sys_combo.currentIndexChanged.connect(self._capture_device_changed)
        sys_row = QVBoxLayout()
        sys_row.addWidget(self.sys_check)
        sys_row.addWidget(self.sys_combo, 1)
        rec_layout.addLayout(sys_row)

        self.mic_verified_check = QCheckBox("Microfono: un parlante verificato")
        self.mic_verified_check.setToolTip("Lascia disattivato se il microfono può riprendere più voci, anche con le cuffie.")
        rec_layout.addWidget(self.mic_verified_check)
        self.echo_check = QCheckBox("Valuta antieco (double-talk)")
        self.echo_check.setToolTip("Usa il loopback come riferimento. Conserva sempre i WAV originali.")
        rec_layout.addWidget(self.echo_check)
        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Ingresso"))
        self.mic_level = QProgressBar()
        self.mic_level.setRange(0, 100)
        self.mic_level.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.mic_level.setTextVisible(False)
        self.mic_level.setAccessibleName("Livello microfono")
        level_row.addWidget(self.mic_level)
        level_row.addWidget(QLabel("Loopback"))
        self.system_level = QProgressBar()
        self.system_level.setRange(0, 100)
        self.system_level.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.system_level.setTextVisible(False)
        self.system_level.setAccessibleName("Livello loopback")
        level_row.addWidget(self.system_level)
        rec_layout.addLayout(level_row)
        device_row = QHBoxLayout()
        self.refresh_devices_btn = QPushButton("Aggiorna")
        self.refresh_devices_btn.clicked.connect(self._populate_devices)
        self.test_devices_btn = QPushButton("Prova dispositivi…")
        self.test_devices_btn.clicked.connect(self._test_devices)
        device_row.addWidget(self.refresh_devices_btn)
        device_row.addWidget(self.test_devices_btn)
        rec_layout.addLayout(device_row)

        self.mp3_check = QCheckBox("Copia MP3 (WAV conservati)")
        self.mp3_check.setChecked(False)
        self.mp3_check.setToolTip(
            "I WAV originali e i metadati di sincronizzazione vengono sempre conservati. "
            "La copia MP3 stereo usa microfono a sinistra e sistema a destra. "
            "La trascrizione continua a usare le sorgenti originali."
        )
        rec_layout.addWidget(self.mp3_check)

        rec_btn_row = QHBoxLayout()
        self.record_btn = QPushButton("Registra")
        self.record_btn.setObjectName("record")
        self.record_btn.setIcon(icon("record", "#ffb8bd"))
        self.record_btn.clicked.connect(self._toggle_record)
        self.elapsed_label = QLabel("00:00")
        self.elapsed_label.setObjectName("timer")
        rec_btn_row.addWidget(self.record_btn)
        rec_btn_row.addWidget(self.elapsed_label)
        rec_layout.addLayout(rec_btn_row)
        audio_page.addWidget(rec_box)

        # --- elaborazione ---
        proc_box = QGroupBox("Impostazioni della trascrizione")
        proc_layout = QVBoxLayout(proc_box)

        self.advanced_options = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_options)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Opzioni avanzate")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.toggled.connect(self.advanced_options.setVisible)
        self.advanced_toggle.toggled.connect(
            lambda expanded: self.advanced_toggle.setArrowType(
                Qt.DownArrow if expanded else Qt.RightArrow
            )
        )
        self.advanced_options.hide()

        opt_row = QHBoxLayout()
        self.denoise_check = QCheckBox("Riduci rumore")
        self.denoise_check.setChecked(False)
        self.denoise_check.setToolTip("Opzionale: confronta prima con l'originale. Il filtraggio può alterare il parlato debole.")
        self.diarize_check = QCheckBox("Distingui le voci")
        self.diarize_check.setChecked(True)
        opt_row.addWidget(self.denoise_check)
        opt_row.addWidget(self.diarize_check)
        advanced_layout.addLayout(opt_row)

        opt_row2 = QVBoxLayout()
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modello trascrizione:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS)
        self.model_combo.setVisible(False)  # configurato dalla schermata Impostazioni
        self.model_summary_label = QLabel()
        self.model_summary_label.setObjectName("muted")
        model_row.addWidget(self.model_summary_label, 1)
        self.model_settings_button = QPushButton("Configura…")
        self.model_settings_button.clicked.connect(self._open_model_settings)
        model_row.addWidget(self.model_settings_button)
        row.addLayout(model_row)
        opt_row2.addWidget(QLabel("Numero di voci"))
        self.speakers_spin = QSpinBox()
        self.speakers_spin.setRange(0, 20)
        self.speakers_spin.setSpecialValueText("Automatico")
        self.speakers_spin.setAccessibleName("Numero di voci, zero per automatico")
        self.speakers_spin.setToolTip(
            "Se conosci il numero di partecipanti, impostalo: con tre o più "
            "voci Tessitura preserva meglio le risposte brevi."
        )
        opt_row2.addWidget(self.speakers_spin)
        opt_row2.addWidget(QLabel("Profilo audio (obbligatorio)"))
        self.audio_profile_combo = QComboBox()
        self.audio_profile_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.audio_profile_combo.setMinimumContentsLength(12)
        self.audio_profile_combo.setAccessibleName("Tipo di registrazione, obbligatorio")
        self.audio_profile_combo.addItem("Seleziona il tipo di registrazione…", None)
        for profile in AudioProfile:
            self.audio_profile_combo.addItem(profile.label, profile.value)
        self.audio_profile_combo.setToolTip(
            "Indica come sono state registrate le voci prima di avviare "
            "trascrizione e diarizzazione. Un recorder stereo ambientale non "
            "ha canali mic/sistema separati."
        )
        self.audio_profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        opt_row2.addWidget(self.audio_profile_combo)
        self.audio_profile_note = QLabel(
            "La scelta è obbligatoria: senza profilo l'elaborazione resta bloccata."
        )
        self.audio_profile_note.setObjectName("muted")
        self.audio_profile_note.setWordWrap(True)
        opt_row2.addWidget(self.audio_profile_note)
        advanced_layout.addWidget(QLabel("Glossario (opzionale)"))
        self.glossary_edit = QLineEdit()
        self.glossary_edit.setPlaceholderText("Nomi e termini separati da virgole")
        self.glossary_edit.setToolTip(
            "Inserisci personaggi, luoghi o parole inventate; vengono usati come "
            "contesto per Whisper senza modificare il testo riconosciuto."
        )
        self.glossary_edit.setAccessibleName("Glossario opzionale, nomi e termini separati da virgole")
        advanced_layout.addWidget(self.glossary_edit)
        opt_row2.addWidget(QLabel("Lingua:"))
        self.lang_combo = QComboBox()
        self.lang_combo.setAccessibleName("Lingua della registrazione")
        for label, code in LANGUAGES:
            self.lang_combo.addItem(label, code)
        self.lang_combo.setToolTip(
            "Whisper usa una lingua sola per registrazione, decisa dai primi "
            "secondi di parlato. Con l'automatico puo' sbagliare (su una lezione "
            "in italiano ha rilevato l'occitano): se la lingua la sai, indicala."
        )
        opt_row2.addWidget(self.lang_combo)
        proc_layout.addLayout(opt_row2)
        proc_layout.addWidget(self.advanced_toggle)
        proc_layout.addWidget(self.advanced_options)

        # L'app si adatta al PC su cui gira: qui si vede cosa ha trovato e cosa usera'.
        self.hw: Hardware = detect()
        idx = self.model_combo.findText(self.hw.model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self._update_model_summary()
        self.hw_label = QLabel(self.hw.describe())
        self.hw_label.setWordWrap(True)
        self.hw_label.setObjectName("muted")
        advanced_layout.addWidget(self.hw_label)

        self.process_btn = QPushButton("Trascrivi registrazione")
        self.process_btn.setObjectName("primary")
        self.process_btn.setIcon(icon("spark", "#102c29"))
        self.process_btn.clicked.connect(self._process_last)
        self.process_btn.setEnabled(False)
        self.process_file_btn = QPushButton("Importa audio…")
        self.process_file_btn.setIcon(icon("import"))
        self.process_file_btn.clicked.connect(self._process_file)
        self.save_text_btn = QPushButton("Esporta testo…")
        self.save_text_btn.clicked.connect(self._save_text)
        self.save_text_btn.setEnabled(False)
        self.save_text_btn.setToolTip("Salva la trascrizione come TXT o SRT")
        self.updates_btn = QPushButton("Aggiornamenti…")
        self.updates_btn.setIcon(icon("refresh"))
        self.updates_btn.clicked.connect(self._open_update_dialog)
        self.updates_btn.setToolTip(
            "Controlla se ci sono versioni piu' recenti delle librerie e dei modelli. "
            "Serve la connessione. Non installa niente."
        )
        self.combine_audio_btn = QPushButton("Combina più audio…")
        self.combine_audio_btn.setIcon(icon("import"))
        self.combine_audio_btn.clicked.connect(self._combine_audio)
        self.combine_audio_btn.setToolTip("Unisce due o più registrazioni in un unico MP3, in sequenza o sovrapposte")
        # Primary actions stay outside scrolling pages and remain discoverable.
        row.addWidget(self.process_file_btn)
        row.addWidget(self.process_btn)
        audio_page.addWidget(self.combine_audio_btn)
        self.diagnostics_btn = QPushButton("Diagnostica audio importato…")
        self.diagnostics_btn.clicked.connect(self._open_diagnostics)
        audio_page.addWidget(self.diagnostics_btn)
        audio_page.addStretch()

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.status_label = QLabel("Pronto per una nuova conversazione.")
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)
        processing_page.addWidget(proc_box)
        processing_page.addStretch()
        project_page.addWidget(self.updates_btn)

        return row

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("File")
        self.open_action = menu.addAction("Apri sessione…", self._open_session)
        self.open_action.setShortcut("Ctrl+O")
        menu.addAction("Salva sessione…", self._save_session).setShortcut("Ctrl+S")
        self.combine_action = menu.addAction(
            "Combina più audio in un MP3…", self._combine_audio
        )
        self.combine_action.setToolTip(
            "Unisce due o più registrazioni in un solo file MP3, in sequenza "
            "oppure sovrapposte."
        )
        menu.addSeparator()
        menu.addAction("Esporta TXT…", lambda: self._export("txt"))
        menu.addAction("Esporta SRT…", lambda: self._export("srt"))
        menu.addAction("Esporta trascrizioni per voce (cartella)…", self._export_speakers_folder)
        self.group_segments_action = menu.addAction(
            "Raggruppa testo per speaker consecutivi…", self._group_segments
        )
        self.group_segments_action.setToolTip(
            "Accorpa i segmenti adiacenti con la stessa voce, mantenendo timestamp e parole."
        )
        self.group_segments_action.setEnabled(False)
        menu.addSeparator()
        menu.addAction("Esci", self.close)

        project_menu = self.menuBar().addMenu("Progetto")
        self.open_project_action = project_menu.addAction(
            "Apri cartella progetto…", self._open_project
        )
        self.open_project_action.setShortcut("Ctrl+Shift+P")
        self.new_project_action = project_menu.addAction(
            "Nuovo progetto in cartella…", self._new_project
        )
        self.project_details_action = project_menu.addAction(
            "Dettagli e memoria…", self._show_project_details
        )
        self.project_details_action.setEnabled(False)
        self.batch_process_action = project_menu.addAction(
            "Elabora tutti gli audio in attesa", self._start_batch_processing
        )
        self.batch_process_action.setEnabled(False)
        project_menu.addSeparator()
        self.close_project_action = project_menu.addAction(
            "Chiudi progetto", self._close_project
        )
        self.close_project_action.setEnabled(False)

        edit_menu = self.menuBar().addMenu("Modifica")
        self.undo_action = edit_menu.addAction("Annulla", self._undo_edit)
        self.undo_action.setShortcut("Ctrl+Z")
        self.redo_action = edit_menu.addAction("Ripeti", self._redo_edit)
        self.redo_action.setShortcut("Ctrl+Y")
        edit_menu.addSeparator()
        self.raw_transcript_action = edit_menu.addAction(
            "Mostra trascrizione grezza", self._toggle_raw_transcript
        )
        self.raw_transcript_action.setCheckable(True)
        self.raw_transcript_action.setToolTip(
            "Confronta l'output prima del controllo automatico delle attribuzioni."
        )
        self.raw_transcript_action.setEnabled(False)

        settings_menu = self.menuBar().addMenu("Impostazioni")
        self.model_settings_action = settings_menu.addAction(
            "Modelli e motori…", self._open_model_settings
        )
        self.first_run_action = settings_menu.addAction(
            "Cartelle dati e configurazione iniziale…", self._open_first_run_dialog
        )
        settings_menu.addSeparator()
        self.updates_action = settings_menu.addAction(
            "Aggiornamenti…", self._open_update_dialog
        )

    def _populate_devices(self) -> None:
        selected = [combo.currentData() for combo in (self.mic_combo, self.sys_combo)]
        try:
            catalog = get_devices()
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Errore dispositivi audio: {exc}")
            return
        for combo, devices, previous, default in (
            (self.mic_combo, catalog.microphones, selected[0], catalog.default_mic),
            (self.sys_combo, catalog.loopbacks, selected[1], catalog.default_loopback),
        ):
            combo.clear()
            for device in devices:
                combo.addItem(device.short(), device)
            preferred = previous or default
            index = next((i for i, device in enumerate(devices)
                          if preferred and device.name == preferred.name
                          and device.host_api == preferred.host_api), -1)
            combo.setCurrentIndex(index)
            if previous is not None and index < 0:
                self.status_label.setText(f"Dispositivo scollegato: {previous.name}. Seleziona un nuovo dispositivo.")

    def _recording_mode_changed(self) -> None:
        self.mic_verified_check.setChecked(False)
        mode = self.recording_mode_combo.currentData()
        self.echo_check.setChecked(mode == "speakers")
        if mode in ("headset", "speakers"):
            self.mic_check.setChecked(True)
            self.sys_check.setChecked(True)

    def _capture_device_changed(self) -> None:
        self.mic_verified_check.setChecked(False)

    def _test_devices(self) -> None:
        from .device_test_dialog import DeviceTestDialog
        dialog = DeviceTestDialog(self.mic_combo.currentData() if self.mic_check.isChecked() else None,
                                  self.sys_combo.currentData() if self.sys_check.isChecked() else None, self)
        dialog.exec()

    def _open_diagnostics(self) -> None:
        source, _ = QFileDialog.getOpenFileName(self, "Diagnostica registrazione lunga", "", "Audio (*.mp3 *.wav *.flac *.m4a)")
        if not source:
            return
        from .diagnostic_dialog import DiagnosticDialog
        dialog = DiagnosticDialog(source, model=self.model_combo.currentText(), device=self.hw.device, parent=self)
        dialog.exec()

    # --------------------------------------------------------------- record
    def _toggle_record(self) -> None:
        if self.recorder.is_recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        if self._busy or self._encoding or self._stopping:
            return
        if not (self.mic_check.isChecked() or self.sys_check.isChecked()):
            QMessageBox.warning(self, "Registrazione", "Seleziona almeno una sorgente.")
            return
        mic_dev: DeviceInfo | None = self.mic_combo.currentData() if self.mic_check.isChecked() else None
        sys_dev: DeviceInfo | None = self.sys_combo.currentData() if self.sys_check.isChecked() else None
        if (self.mic_check.isChecked() and mic_dev is None) or (self.sys_check.isChecked() and sys_dev is None):
            QMessageBox.warning(self, "Registrazione", "Seleziona i dispositivi disponibili. Usa Aggiorna dispositivi dopo una disconnessione.")
            return
        config = dict(recording_mode=self.recording_mode_combo.currentData(),
                      mic_single_speaker_verified=self.mic_verified_check.isChecked(),
                      echo_cancellation=self.echo_check.isChecked())
        try:
            self.recorder.start(
                record_mic=self.mic_check.isChecked(),
                record_system=self.sys_check.isChecked(),
                mic_device=mic_dev,
                system_device=sys_dev,
                **config,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Registrazione", f"Impossibile avviare: {exc}")
            return
        self._last_recording_config = config
        self._pipeline_strategy = "source_aware" if self.sys_check.isChecked() and self.mic_check.isChecked() else "standard"
        self._headphones_confirmed = config["recording_mode"] == "headset"
        self._pipeline_strategy_configured = True
        self._elapsed = 0
        self._record_started = time.monotonic()
        self.elapsed_label.setText("00:00")
        self._timer.start()
        self.record_btn.setText("Ferma")
        self.record_btn.setIcon(icon("stop", "#ffb8bd"))
        self._set_busy(False)
        self.status_label.setText("Registrazione in corso…")

    def _stop_record(self) -> None:
        if self._stopping or self._stopper is not None:
            return
        self._timer.stop()
        self._stopping = True
        self._set_busy(False)
        self.status_label.setText("Finalizzazione della registrazione…")
        self._stopper = StopRecordWorker(self.recorder, self)
        self._stopper.done.connect(self._on_record_stopped)
        self._stopper.failed.connect(self._on_stop_failed)
        self._stopper.finished.connect(self._stopper.deleteLater)
        self._stopper.finished.connect(lambda: setattr(self, "_stopper", None))
        self._stopper.start()

    def _on_stop_failed(self, message: str) -> None:
        self._stopping = False
        self._set_busy(False)
        self.status_label.setText("Registrazione non finalizzata. I file WAV sono stati conservati.")
        QMessageBox.critical(self, "Registrazione", message)

    def _on_record_stopped(self, result) -> None:
        self._stopping = False
        self.record_btn.setText("Registra")
        self.record_btn.setIcon(icon("record", "#ffb8bd"))
        self.last_mic = result.mic_path
        self.last_system = result.system_path
        self.last_offsets = result.offsets()
        self.last_dual = False
        metadata_path = getattr(result, "metadata_path", None)
        if metadata_path is not None:
            import json
            try:
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                self._last_recording_config = dict(
                    recording_mode=result.recording_mode,
                    mic_single_speaker_verified=result.mic_single_speaker_verified,
                    echo_cancellation=result.echo_cancellation,
                    recording_metadata=metadata,
                )
            except (OSError, ValueError):
                pass  # The original tracks remain usable without a sidecar.
        self._set_audio_profile(
            AudioProfile.MIC_SYSTEM
            if self.last_mic is not None and self.last_system is not None
            else AudioProfile.STEREO_AMBIENT
        )
        msg = "Registrazione salvata."
        if result.errors:
            msg += " Attenzione: " + "; ".join(result.errors)

        if self.mp3_check.isChecked() and (self.last_mic or self.last_system):
            self._encoding = True
            self._set_busy(False)
            self.status_label.setText(msg + " Unione in MP3…")
            self._encoder = EncodeWorker(
                self.last_mic, self.last_system, self.last_offsets, self
            )
            self._encoder.done.connect(self._on_encoded)
            self._encoder.finished.connect(self._encoder.deleteLater)
            self._encoder.finished.connect(lambda: setattr(self, "_encoder", None))
            self._encoder.start()
            return

        self._set_busy(False)
        self.status_label.setText(msg)

    def _on_encoded(self, merged_path, dual: bool, warning: str) -> None:
        if merged_path is not None:
            # The MP3 is an export. Keep original source paths and offsets for ASR.
            self.last_mp3_export = merged_path
        self._encoding = False
        self._set_busy(False)
        self.status_label.setText(
            f"Registrazione salvata (WAV). {warning}" if warning
            else f"Copia MP3 salvata: {merged_path.name}. WAV originali conservati."
        )

    def _tick(self) -> None:
        self._elapsed = int(time.monotonic() - self._record_started)
        self.elapsed_label.setText(f"{self._elapsed // 60:02d}:{self._elapsed % 60:02d}")
        levels = self.recorder.levels()
        self.mic_level.setValue(min(100, int(levels["mic"] * 300)))
        self.system_level.setValue(min(100, int(levels["system"] * 300)))
        if self.recorder.capture_errors():
            self.status_label.setText("Errore dispositivo: arresto e conservazione delle tracce originali…")
            self._stop_record()
        elif self.sys_check.isChecked() and self._elapsed >= 5 and levels["system"] < 0.001:
            self.status_label.setText("Loopback silenzioso: verifica che l'applicazione riproduca sull'uscita selezionata.")

    # -------------------------------------------------------------- process
    def _build_options(self, *, for_recording: bool = True) -> PipelineOptions:
        n = self.speakers_spin.value()
        profile = AudioProfile.parse(self.audio_profile_combo.currentData())
        if profile is None:
            raise ValueError(
                "Seleziona il profilo audio prima di avviare l'elaborazione."
            )
        typed_glossary = self.glossary_edit.text().strip()
        initial_prompt = combined_glossary(self.project, typed_glossary)
        known_profiles = (
            self.project.speaker_profiles()
            if self.project is not None and getattr(self.project, "learning_enabled", False)
            else None
        )
        return PipelineOptions(
            denoise=self.denoise_check.isChecked(),
            diarize=self.diarize_check.isChecked(),
            language=self.lang_combo.currentData(),
            model_size=self.model_combo.currentText(),
            device=self.hw.device,
            compute_type=self.hw.compute_type,
            num_speakers=n or None,
            audio_profile=profile,
            initial_prompt=initial_prompt,
            analysis_mode="safe",
            known_speaker_profiles=known_profiles,
            strategy=getattr(self, "_pipeline_strategy", "standard") if for_recording else "standard",
            headphones_confirmed=getattr(self, "_headphones_confirmed", True) if for_recording else None,
            **((self._last_recording_config or {}) if for_recording else {}),
        )

    def _on_profile_changed(self, _index: int) -> None:
        profile = AudioProfile.parse(self.audio_profile_combo.currentData())
        if profile is None:
            self.audio_profile_note.setText(
                "La scelta è obbligatoria: senza profilo l'elaborazione resta bloccata."
            )
        else:
            self.audio_profile_note.setText(profile.description)
        if hasattr(self, "_busy"):
            self._set_busy(self._busy)
        if hasattr(self, "_update_project_ui"):
            self._update_project_ui()

    def _set_audio_profile(self, profile: AudioProfile | None) -> None:
        value = profile.value if profile is not None else None
        index = self.audio_profile_combo.findData(value)
        self.audio_profile_combo.setCurrentIndex(max(0, index))
        if hasattr(self, "_update_project_ui"):
            self._update_project_ui()

    def _process_last(self) -> None:
        if not (self.last_mic or self.last_system):
            QMessageBox.information(self, "Elaborazione", "Nessuna registrazione da elaborare.")
            return
        self._run_pipeline(
            self.last_mic, self.last_system, self.last_offsets, self.last_dual
        )

    def _process_file(self) -> None:
        selected_paths, _ = QFileDialog.getOpenFileNames(
            self, "Scegli uno o due file audio (seleziona 2 per traccia Mic + Sistema)", "",
            "Audio (*.wav *.flac *.mp3 *.m4a *.ogg *.opus *.aac *.wma)"
        )
        if not selected_paths:
            return
        if not self._confirm_discard():
            return
        self._last_recording_config = None
        self._pipeline_strategy = "standard"
        self._pipeline_strategy_configured = False
        if len(selected_paths) <= 2:
            from ..audio.recording_metadata import load_recording_metadata, recording_sources
            try:
                metadata = load_recording_metadata(selected_paths[0])
                if metadata is not None:
                    mic, system, offsets = recording_sources(selected_paths[0], metadata)
                    if len(selected_paths) == 2 and {Path(path).resolve() for path in selected_paths} != {path for path in (mic, system) if path is not None}:
                        raise ValueError("I due file selezionati non sono le sorgenti della stessa acquisizione")
                    self.last_mic, self.last_system, self.last_offsets = mic, system, offsets
                    self.last_dual = False
                    self._last_recording_config = dict(
                        recording_mode=metadata["recording_mode"],
                        mic_single_speaker_verified=metadata.get("mic_single_speaker_verified", False),
                        echo_cancellation=metadata.get("echo_cancellation", False),
                        recording_metadata=metadata,
                    )
                    self._pipeline_strategy = "source_aware" if self.last_mic and self.last_system else "standard"
                    self._headphones_confirmed = metadata["recording_mode"] == "headset"
                    self._pipeline_strategy_configured = True
                    self._set_audio_profile(AudioProfile.MIC_SYSTEM if self.last_mic and self.last_system else AudioProfile.STEREO_AMBIENT)
                    self._set_busy(False)
                    self.status_label.setText("Sorgenti originali e sincronizzazione ripristinate dai metadati di acquisizione.")
                    return
            except (OSError, ValueError, TypeError) as exc:
                QMessageBox.warning(self, "Metadati di acquisizione", f"Impossibile ripristinare le sorgenti: {exc}. Il file verrà importato senza ipotesi sulle voci.")
        if len(selected_paths) == 1:
            self.last_mic = Path(selected_paths[0])
            self.last_system = None
            self.last_offsets = []
            self.last_dual = False
            self._set_audio_profile(None)
            self._set_busy(False)
            self.status_label.setText(
                "Audio importato. Seleziona il profilo audio, poi avvia la trascrizione."
            )
        else:
            self.last_mic = Path(selected_paths[0])
            self.last_system = Path(selected_paths[1])
            self.last_offsets = [0.0, 0.0]
            self.last_dual = False
            self._set_audio_profile(AudioProfile.MIC_SYSTEM)
            self._set_busy(False)
            strategy, headphones = ask_recording_mode(self)
            self._pipeline_strategy = strategy
            self._headphones_confirmed = headphones
            self._pipeline_strategy_configured = True
            self.status_label.setText(
                f"Due tracce importate ({self.last_mic.name}, {self.last_system.name}). Premi Trascrivi per avviare."
            )

    def _combine_audio(self) -> None:
        """Unisce due o piu registrazioni in un solo MP3, su scelta dell'utente."""
        if self._busy or self._encoding or self._stopping or self._combiner is not None:
            self.status_label.setText("Attendi il termine dell'operazione in corso.")
            return
        selected_paths, _ = QFileDialog.getOpenFileNames(
            self, "Scegli i file audio da combinare (almeno due)",
            str(paths.recordings_dir()),
            "Audio (*.wav *.flac *.mp3 *.m4a *.ogg *.opus *.aac *.wma)",
        )
        if not selected_paths:
            return
        if len(selected_paths) < 2:
            QMessageBox.information(
                self, "Combina audio",
                "Seleziona almeno due file: con uno solo non c'è niente da unire.",
            )
            return

        box = QMessageBox(self)
        box.setWindowTitle("Combina audio")
        box.setIcon(QMessageBox.Question)
        box.setText(f"{len(selected_paths)} file selezionati. Come vanno combinati?")
        box.setInformativeText(
            "In sequenza: i file vengono messi uno dopo l'altro, nell'ordine "
            "scelto. È il caso di una registrazione spezzata in più parti.\n\n"
            "Sovrapposti: i file suonano insieme, come due microfoni sulla "
            "stessa scena. La durata è quella del file più lungo."
        )
        sequence = box.addButton("In sequenza", QMessageBox.AcceptRole)
        overlay = box.addButton("Sovrapposti", QMessageBox.AcceptRole)
        box.addButton("Annulla", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (sequence, overlay):
            return
        mode = "concat" if clicked is sequence else "mix"

        stem = Path(selected_paths[0]).stem
        suffix = "unione" if mode == "concat" else "sovrapposizione"
        default = paths.recordings_dir() / f"{stem}_{suffix}.mp3"
        destination, _ = QFileDialog.getSaveFileName(
            self, "Salva l'MP3 combinato", str(default), "MP3 (*.mp3)"
        )
        if not destination:
            return
        target = Path(destination)
        if target.suffix.lower() != ".mp3":
            target = target.with_suffix(".mp3")

        self._combiner = CombineAudioWorker(selected_paths, target, mode=mode, parent=self)
        self._combiner.progress.connect(self._on_progress)
        self._combiner.done.connect(self._on_audio_combined)
        self._combiner.failed.connect(self._on_combine_failed)
        self._combiner.finished.connect(self._combiner.deleteLater)
        self._combiner.finished.connect(self._on_combine_finished)
        self.progress.setValue(0)
        self.status_label.setText(f"Combinazione di {len(selected_paths)} file in corso…")
        self._set_busy(True)
        self._combiner.start()

    def _on_combine_finished(self) -> None:
        self._combiner = None
        self._set_busy(False)

    def _on_audio_combined(self, destination) -> None:
        path = Path(destination)
        self.progress.setValue(100)
        self.status_label.setText(f"MP3 combinato: {path.name}")
        answer = QMessageBox.question(
            self, "Combina audio",
            f"File creato: {path.name}\n\nVuoi usarlo subito per la trascrizione?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        if not self._confirm_discard():
            return
        self.last_mic = path
        self.last_system = None
        self.last_offsets = []
        self.last_dual = False
        self._last_recording_config = None
        self._pipeline_strategy = "standard"
        self._pipeline_strategy_configured = False
        self._set_audio_profile(None)
        self._set_busy(False)
        self.status_label.setText(
            "Audio combinato pronto. Seleziona il profilo audio, poi avvia la trascrizione."
        )

    def _on_combine_failed(self, message: str) -> None:
        self.progress.setValue(0)
        self.status_label.setText("Combinazione non riuscita.")
        QMessageBox.critical(self, "Combina audio", message)

    def _run_pipeline(self, mic_path, system_path, offsets=None, dual=False) -> None:
        if self._busy or self._encoding or self._stopping or self.recorder.is_recording or self._worker is not None:
            return
        if AudioProfile.parse(self.audio_profile_combo.currentData()) is None:
            QMessageBox.warning(
                self,
                "Profilo audio obbligatorio",
                "Indica se il file è 'Microfono + audio di sistema' oppure "
                "'Ambientale stereo' prima di trascrivere e distinguere le voci.",
            )
            return
        if not self._confirm_discard():
            return
        self.player.clear()
        self.progress.setValue(0)
        self._set_busy(True)
        self.status_label.setText(self._estimate(mic_path or system_path))
        self._worker = ProcessWorker(
            mic_path, system_path, self._build_options(), offsets_s=offsets or [],
            dual_channel=dual,
        )
        self._worker.progress.connect(self._on_progress)
        self.stage_toggle.show()
        self.stage_toggle.setChecked(True)
        self.stage_view.begin_run()
        self._worker.stage.connect(self.stage_view.on_stage)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))
        self._worker.start()

    def _estimate(self, audio_path) -> str:
        """Quanto ci vorra' su questa macchina, detto prima di iniziare."""
        model = self.model_combo.currentText()
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
            seconds = info.frames / float(info.samplerate or 1)
        except Exception:  # noqa: BLE001 - la stima e' un di piu'
            return "Elaborazione avviata…"
        return estimate_text(self.hw, seconds, model)

    def _on_progress(self, message: str, frac: float) -> None:
        self.progress.setValue(int(frac * 100))
        self.status_label.setText(message)

    def _on_done(self, session: Session) -> None:
        self.session = session
        self.session_path = None
        if self.project is not None and self.last_mic is not None:
            tag_session(self.project, session)
            apply_project_names(self.project, session)
            summary = summary_for_memory(session)
            store_session(self.project, self.last_mic, session, summary=summary)
            self._update_project_ui()
        if session.model:
            index = self.model_combo.findText(session.model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                self._update_model_summary()
        self._showing_raw = False
        self.raw_transcript_action.setChecked(False)
        self._display_transcript()
        self.speakers.set_session(session)
        self._describe_session()
        self._mark_dirty()
        self._load_audio(session)
        self._set_busy(False)
        self.progress.setValue(100)
        prefix = f"{self.project.name} · " if self.project is not None else ""
        self.status_label.setText(
            f"{prefix}Completato: {len(session.segments)} segmenti, "
            f"{len(session.speaker_ids())} voci (lingua: {session.language})."
        )
        synchronization = (session.capture_manifest or {}).get("synchronization", {})
        if synchronization.get("review_required"):
            self.status_label.setText(self.status_label.text() + " Sincronizzazione incerta: verifica le tracce originali.")

    def _on_active_view_changed(self, view_name: str) -> None:
        if view_name == "timeline":
            self.transcript_stack.setCurrentWidget(self.timeline_view)
            self.timeline_view_btn.setChecked(True)
            self._previous_speaker_sidebar_visible = self.speaker_sidebar.isVisible()
            self.voices_toggle.setChecked(False)
            self.speaker_sidebar.setVisible(False)
        elif view_name == "table":
            self.transcript_stack.setCurrentWidget(self.transcript)
            self.table_view_btn.setChecked(True)
            if getattr(self, "_previous_speaker_sidebar_visible", True):
                self.voices_toggle.setChecked(True)
                self.speaker_sidebar.setVisible(True)
        else:
            self.transcript_stack.setCurrentWidget(self.conversation_view)
            self.conv_view_btn.setChecked(True)
            if getattr(self, "_previous_speaker_sidebar_visible", True):
                self.voices_toggle.setChecked(True)
                self.speaker_sidebar.setVisible(True)

    def _on_player_position_changed(self, seconds: float) -> None:
        self.controller.update_playback_position(seconds)
        self.timeline_view.update_playback_position(seconds)
        self.transcript.highlight_at(seconds)

    def _on_speaker_names_changed(self) -> None:
        if self.session:
            self.controller.sessionChanged.emit(self.session)
            self.controller.speakerNamesChanged.emit(self.session.speaker_names)

    def _display_transcript(self) -> None:
        """Mostra la versione revisionata oppure quella grezza, senza sostituire i dati."""
        if self.session is None:
            return
        showing_raw = self._showing_raw and self.session.raw_segments is not None
        raw_utts = getattr(self.session, "raw_utterances", None)
        if showing_raw:
            from copy import deepcopy
            from ..processing.merge import build_utterances_from_segments
            raw_utts = deepcopy(raw_utts) if raw_utts else build_utterances_from_segments(deepcopy(self.session.raw_segments))
        displayed = replace(
            self.session,
            segments=self.session.raw_segments if showing_raw else self.session.segments,
            utterances=raw_utts if (showing_raw and raw_utts) else self.session.utterances,
            schema_version=6,
        ) if showing_raw else self.session
        self.controller.set_read_only(showing_raw)
        self.controller.load_session(displayed)
        self.transcript.set_session(displayed)
        self.transcript.set_read_only(showing_raw)
        if hasattr(displayed, "active_view") and displayed.active_view in TranscriptController.SUPPORTED_VIEWS:
            self.controller.set_active_view(displayed.active_view)
        if showing_raw:
            self.review_hint.setText(
                f"Versione grezza · {len(displayed.segments)} segmenti · "
                "le attribuzioni non sono state revisionate."
            )
        else:
            self._describe_session()

    def _on_timeline_changed(self) -> None:
        if self.session is None or self._showing_raw:
            return
        self._mark_dirty()
        self.transcript.set_session(self.session)
        self.speakers.set_session(self.session)

    def _refresh_canonical_views(self, *_args) -> None:
        if self.session and not self._showing_raw:
            self.controller.sessionChanged.emit(self.session)

    def _reprocess_region(self, start_s: float, end_s: float, separate: bool) -> None:
        if self.session is None or self._busy or self._showing_raw or getattr(self, "_region_worker", None) is not None:
            return
        source = paths.resolve_media(self.session.audio.mixed or self.session.audio.primary())
        if source is None:
            QMessageBox.information(self, "Recupero regione", "Audio non disponibile.")
            return
        from copy import deepcopy
        from .region_worker import RegionRecoveryWorker
        session_id = self.session.id
        potential = max((span.get("potential_speaker_count", 1) for span in self._region_activity(start_s, end_s)), default=None)
        options = {"separate": separate, "potential_speakers": potential, "existing_events": deepcopy(self.session.utterances),
                   "model_size": self.session.model or "large-v3", "language": self.session.language}
        self._region_worker = RegionRecoveryWorker(source, start_s, end_s, paths.work_dir(), options=options, parent=self)
        self._set_busy(True)
        self.status_label.setText("Recupero della regione selezionata: i risultati saranno proposte da revisionare.")

        def completed(result):
            if self.session and self.session.id == session_id:
                self.session.begin_edit("Proposte recupero regione")
                self.session.recovery_candidates.extend(result["candidates"])
                self.controller._notify_changed()
                self.status_label.setText(f"{len(result['candidates'])} candidati da revisionare. Separazione: {result['separation']['status']}.")

        def failed(message):
            self.status_label.setText(f"Recupero non completato: {message}")

        def finished():
            worker = self._region_worker
            self._region_worker = None
            self._set_busy(False)
            worker.deleteLater()

        self._region_worker.done.connect(completed)
        self._region_worker.failed.connect(failed)
        self._region_worker.finished.connect(finished)
        self._region_worker.start()

    def _region_activity(self, start_s, end_s):
        from ..processing.global_diarization import activity_spans
        return [s for s in activity_spans(self.session.diarization.get("turns", [])) if s["end_s"] > start_s and s["start_s"] < end_s]

    def _sync_audio_source(self, source: str) -> None:
        self.timeline_view.set_audio_path(source)
        if not self.session:
            return
        original = paths.resolve_media(self.session.audio.mic)
        processed = paths.resolve_media(self.session.audio.mixed)
        combo = self.timeline_view.audio_variant
        actual = Path(source).resolve()
        variant = "original" if original and actual == original.resolve() else "processed"
        combo.blockSignals(True)
        combo.setCurrentIndex(combo.findData(variant))
        combo.setEnabled(bool(original and processed and self.session.audio_profile == "stereo_ambient"))
        combo.blockSignals(False)

    def _review_global_identities(self) -> None:
        if self.session is None or self._busy or self._showing_raw or getattr(self, "_identity_worker", None) is not None:
            return
        from copy import deepcopy
        from .identity_worker import GlobalIdentityWorker
        from ..processing.global_diarization import annotate_identity_uncertainty
        original = self.session.audio.mic if self.session.audio_profile == "stereo_ambient" else self.session.audio.mixed
        source = paths.resolve_media(original or self.session.audio.primary())
        if source is None:
            self.status_label.setText("Audio non disponibile per la revisione globale.")
            return
        fingerprint = self.session.fingerprint()
        worker = GlobalIdentityWorker(deepcopy(self.session), source, paths.cache_dir() / "global-identity" / self.session.id, self)
        self._identity_worker = worker
        self._set_busy(True)
        self.status_label.setText("Revisione identità nell'intero audio con campioni puliti e conferme manuali…")

        def completed(result):
            if not self.session or self.session.fingerprint() != fingerprint:
                self.status_label.setText("La trascrizione è cambiata: riesegui la revisione globale. Le correzioni sono conservate.")
                return
            self.session.begin_edit("Revisione globale delle identità")
            annotate_identity_uncertainty(self.session.utterances, result["review"])
            self.session.diarization["global_identity_review"] = result["review"]
            known = {c["candidate_id"] for c in self.session.recovery_candidates if c.get("status") != "proposed"}
            self.session.recovery_candidates = [c for c in self.session.recovery_candidates if c.get("kind") != "speaker_assignment" or c.get("status") != "proposed"]
            self.session.recovery_candidates.extend(c for c in result["candidates"] if c["candidate_id"] not in known)
            self.controller._notify_changed()
            short = len(result["review"].get("confirmed_identity_review", {}).get("suspected_regions", []))
            self.status_label.setText(f"{len(result['candidates'])} proposte globali; {short} conferme richiedono campioni puliti più lunghi.")

        def finished():
            self._identity_worker = None
            self._set_busy(False)
            worker.deleteLater()

        worker.done.connect(completed)
        worker.failed.connect(lambda message: self.status_label.setText(f"Revisione globale non completata: {message}"))
        worker.finished.connect(finished)
        worker.start()

    def _toggle_raw_transcript(self, checked: bool) -> None:
        if self.session is None or (self.session.raw_segments is None and getattr(self.session, "raw_utterances", None) is None):
            self.raw_transcript_action.setChecked(False)
            return
        self._showing_raw = bool(checked)
        self._display_transcript()

    def _load_audio(self, session: Session) -> bool:
        """Carica nel player l'audio della sessione, ritrovandolo se e' stato spostato."""
        found = session.audio.resolved_primary()
        if found is None:
            self.player.clear()
            return False
        self.player.load_session_audio(session)
        return True

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.progress.setValue(0)
        self.status_label.setText("Errore durante l'elaborazione.")
        QMessageBox.critical(self, "Elaborazione", message)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        recording = self.recorder.is_recording
        idle = not (busy or self._encoding or recording or self._stopping)
        self._idle = idle
        profile_selected = (
            hasattr(self, "audio_profile_combo")
            and AudioProfile.parse(self.audio_profile_combo.currentData()) is not None
        )
        self.process_btn.setEnabled(
            idle and bool(self.last_mic or self.last_system) and profile_selected
        )
        self.process_file_btn.setEnabled(idle)
        if hasattr(self, "combine_action"):
            self.combine_action.setEnabled(idle)
        if hasattr(self, "combine_audio_btn"):
            self.combine_audio_btn.setEnabled(idle)
        if hasattr(self, "open_project_btn"):
            self.open_project_btn.setEnabled(idle)
        if hasattr(self, "open_project_action"):
            self.open_project_action.setEnabled(idle)
        if hasattr(self, "new_project_action"):
            self.new_project_action.setEnabled(idle)
        self.save_text_btn.setEnabled(self.session is not None)
        self.record_btn.setEnabled(not (busy or self._encoding or self._stopping))
        self.open_action.setEnabled(idle)
        self.model_settings_action.setEnabled(idle)
        self._update_edit_actions()
        for widget in (self.mic_check, self.sys_check, self.mic_combo, self.sys_combo,
                       self.mp3_check, self.model_combo, self.lang_combo,
                       self.denoise_check, self.diarize_check,
                       self.speakers_spin,
                       self.audio_profile_combo, self.glossary_edit,
                       self.model_settings_button):
            widget.setEnabled(idle)
        for widget in (self.recording_mode_combo, self.mic_verified_check, self.echo_check,
                       self.refresh_devices_btn, self.test_devices_btn, self.diagnostics_btn):
            widget.setEnabled(idle)
        self._update_session_actions()
        if hasattr(self, "_update_project_ui"):
            self._update_project_ui()

    def _describe_session(self) -> None:
        self.session_title.setText(self.session.name)
        profile = AudioProfile.parse(self.session.audio_profile)
        profile_text = f" · {profile.label}" if profile is not None else ""
        review = self.session.transcript_review or {}
        ambiguous = int(review.get("ambiguous_words", 0) or 0)
        review_text = f" · {ambiguous} parole da verificare" if ambiguous else ""
        project_text = ""
        prov = getattr(self.session, "provenance", None)
        if isinstance(prov, dict) and prov.get("project"):
            p_name = prov["project"].get("name")
            if p_name:
                project_text = f" · Progetto: {p_name}"
        elif self.project is not None:
            project_text = f" · Progetto: {self.project.name}"
        self.review_hint.setText(
            f"{len(self.session.segments)} segmenti · {len(self.session.speaker_ids())} voci"
            f"{profile_text}{review_text}{project_text} · Clicca il tempo per ascoltare; "
            "doppio clic sul testo per modificarlo."
        )

    # ----------------------------------------------------------- project ops
    def _open_project(self, folder_path: str | Path | None = None) -> None:
        if not folder_path:
            folder = QFileDialog.getExistingDirectory(
                self, "Scegli la cartella del progetto",
                str(paths.data_dir()),
            )
            if not folder:
                return
            folder_path = Path(folder)
        else:
            folder_path = Path(folder_path)

        try:
            self.project = open_or_create(folder_path)
            self._update_project_ui()
            pending = len(self.project.pending_audio())
            self.status_label.setText(
                f"Progetto aperto: {self.project.name} ({pending} audio in attesa)"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Progetto", f"Impossibile aprire il progetto:\n{exc}")

    def _new_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Scegli la cartella per il nuovo progetto",
            str(paths.data_dir()),
        )
        if not folder:
            return
        folder_path = Path(folder)
        self._open_project(folder_path)
        if self.project:
            self._show_project_details()

    def _show_project_details(self) -> None:
        if self.project is None:
            return
        dialog = ProjectDetailsDialog(self.project, parent=self)
        if dialog.exec() == QDialog.Accepted:
            dialog.apply_to_project()
            self.project.save()
            self._update_project_ui()
            self.status_label.setText(f"Dati progetto '{self.project.name}' aggiornati.")

    def _close_project(self) -> None:
        if self.project is None:
            return
        name = self.project.name
        self.project = None
        self._update_project_ui()
        self.status_label.setText(f"Progetto '{name}' chiuso.")

    def _update_project_ui(self) -> None:
        idle = getattr(self, "_idle", True)
        batch_running = self._batch_worker is not None and self._batch_worker.isRunning()
        has_profile = (
            hasattr(self, "audio_profile_combo")
            and AudioProfile.parse(self.audio_profile_combo.currentData()) is not None
        )

        if self.project is None:
            if hasattr(self, "project_label"):
                self.project_label.setText("Nessun progetto aperto")
                self.project_status.setText("Apri una cartella per collegare audio, memoria e glossario.")
                self.project_details_btn.setEnabled(False)
                self.batch_process_btn.setEnabled(False)
                self.batch_process_btn.setText("Elabora coda audio")
            if hasattr(self, "project_details_action"):
                self.project_details_action.setEnabled(False)
                self.batch_process_action.setEnabled(False)
                self.close_project_action.setEnabled(False)
            return

        pending = self.project.pending_audio()
        done = len(self.project.sessions)
        total = len(self.project.audio_files())

        if hasattr(self, "project_label"):
            self.project_label.setText(f"📁 {self.project.name}")
            self.project_status.setText(
                f"{done} sedute in memoria · {len(pending)}/{total} audio da elaborare"
            )
            self.project_details_btn.setEnabled(idle and not batch_running)
            if batch_running:
                self.batch_process_btn.setEnabled(not self._batch_worker.stopping)
                self.batch_process_btn.setText(
                    "Ferma coda" if not self._batch_worker.stopping else "Arresto in corso…"
                )
            else:
                can_batch = idle and bool(pending) and has_profile
                self.batch_process_btn.setEnabled(can_batch)
                self.batch_process_btn.setText(
                    f"Elabora coda ({len(pending)})" if pending else "Tutti gli audio elaborati"
                )

        if hasattr(self, "project_details_action"):
            self.project_details_action.setEnabled(idle and not batch_running)
            self.batch_process_action.setEnabled(
                idle and bool(pending) and has_profile and not batch_running
            )
            self.close_project_action.setEnabled(idle and not batch_running)

    def _start_batch_processing(self) -> None:
        if self.project is None:
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_worker.stop()
            self.status_label.setText("Arresto della coda dopo il file in corso…")
            self.batch_process_btn.setEnabled(False)
            return

        pending = self.project.pending_audio()
        if not pending:
            QMessageBox.information(
                self, "Elaborazione coda",
                "Tutti i file audio nella cartella del progetto sono già stati elaborati."
            )
            return

        try:
            options = self._build_options(for_recording=False)
        except ValueError as err:
            QMessageBox.warning(self, "Profilo audio obbligatorio", str(err))
            return

        self._batch_worker = BatchProcessWorker(
            pending, options, parent=self, project_name=self.project.name
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self.stage_toggle.show()
        self.stage_toggle.setChecked(True)
        self.stage_view.begin_run()
        self._batch_worker.stage.connect(self.stage_view.on_batch_stage)
        self._batch_worker.file_done.connect(self._on_batch_file_done)
        self._batch_worker.file_failed.connect(self._on_batch_file_failed)
        self._batch_worker.done.connect(self._on_batch_done)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self._set_busy(True)
        self.progress.setValue(0)
        self.status_label.setText(
            f"{self.project.name} · Avvio coda di {len(pending)} file…"
        )
        self._batch_worker.start()
        self._update_project_ui()

    def _on_batch_progress(self, message: str, frac: float) -> None:
        self.progress.setValue(int(frac * 100))
        self.status_label.setText(message)

    def _on_batch_file_done(self, audio: Path, session: Session) -> None:
        if self.project is not None:
            tag_session(self.project, session)
            apply_project_names(self.project, session)
            summary = summary_for_memory(session)
            store_session(self.project, audio, session, summary=summary)
            self._update_project_ui()
        self.session = session
        self.session_path = None
        self.last_mic = audio
        self.last_system = None
        self.last_offsets = []
        self.last_dual = False
        self._showing_raw = False
        self.raw_transcript_action.setChecked(False)
        self._display_transcript()
        self.speakers.set_session(session)
        self._describe_session()
        self._load_audio(session)
        self._mark_dirty()

    def _on_batch_file_failed(self, audio: Path, error: str) -> None:
        prefix = f"{self.project.name} · " if self.project else ""
        self.status_label.setText(f"{prefix}File {audio.name} non riuscito: {error}")

    def _on_batch_done(self, completed: list[Path]) -> None:
        stopped = self._batch_worker is not None and self._batch_worker.stopping
        if not stopped:
            self.progress.setValue(100)
        proj_name = self.project.name if self.project else ""
        outcome = "Coda arrestata" if stopped else "Coda completata"
        self.status_label.setText(f"{proj_name} · {outcome}: {len(completed)} file elaborati.")
        QMessageBox.information(
            self, outcome,
            f"Elaborati con successo {len(completed)} file nel progetto '{proj_name}'.",
        )

    def _on_batch_failed(self, error: str) -> None:
        self.progress.setValue(0)
        self.status_label.setText("Errore durante l'elaborazione della coda.")
        QMessageBox.critical(self, "Errore coda", error)

    def _on_batch_finished(self) -> None:
        self._batch_worker = None
        self._set_busy(False)
        self._update_project_ui()

    def _on_speaker_reassigned(self, _row: int, _speaker_id: str) -> None:
        """Correzione manuale dell'attribuzione: la voce puo' restare senza frasi."""
        self._mark_dirty()
        if self.session is not None:
            self.speakers.set_session(self.session)

    def _on_speaker_name_confirmed(self, speaker_id: str, name: str) -> None:
        if self.session is None:
            return
        if name:
            confirm_speakers(self.session, [speaker_id])
        else:
            unconfirm_speaker(self.session, speaker_id)
        self._mark_dirty()
        if name and self.project is not None and self.project.learning_enabled:
            self.status_label.setText(
                f"Identità di {name} confermata; il profilo verrà appreso al salvataggio."
            )

    def _sync_project_learning(self) -> dict[str, int | bool | str]:
        if self.project is None or self.session is None:
            return {"changed": False, "corrections": 0, "speaker_profiles": 0}
        try:
            report = sync_project_learning(self.project, self.session)
            if report.get("changed"):
                self._update_project_ui()
            return report
        except Exception as exc:  # noqa: BLE001 - la sessione resta modificabile/salvabile
            self.status_label.setText(
                f"Modifica conservata nella sessione; memoria progetto non aggiornata: {exc}"
            )
            return {
                "changed": False, "corrections": 0, "speaker_profiles": 0,
                "error": str(exc),
            }

    def _update_session_actions(self) -> None:
        """Voci di menu che dipendono da com'e' fatta la sessione adesso.

        Il raggruppamento era calcolato solo dentro _set_busy, cioe' agli
        estremi di un'elaborazione: dopo una riassegnazione manuale di voce la
        situazione cambia (due segmenti contigui diventano della stessa persona)
        ma il menu restava com'era, quindi la funzione risultava indisponibile
        proprio quando serviva.
        """
        if not hasattr(self, "group_segments_action"):
            return
        idle = getattr(self, "_idle", True)
        session = self.session
        has_segments = session is not None and bool(session.segments)
        self.raw_transcript_action.setEnabled(
            idle and has_segments and session.raw_segments is not None
        )
        self.group_segments_action.setEnabled(
            idle and has_segments and session.adjacent_same_speaker_groups() > 0
        )

    def _mark_dirty(self, *_args) -> None:
        self._dirty = True
        self.setWindowTitle("Tessitura — sessione modificata *")
        self._update_edit_actions()
        self._update_session_actions()

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self, "Sessione non salvata", "Salvare la sessione prima di continuare?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Save:
            self._save_session()
            return not self._dirty
        return answer == QMessageBox.Discard

    # ------------------------------------------------------------- file ops
    def _open_session(self) -> None:
        if self._busy or self._encoding or self._stopping or self.recorder.is_recording:
            return
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Apri sessione", str(paths.sessions_dir()), "JSON (*.json)"
        )
        if not path:
            return
        try:
            session = storage.load_session(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Apri sessione", str(exc))
            return
        self.session = session
        self.session_path = Path(path)
        if session.model:
            index = self.model_combo.findText(session.model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                self._update_model_summary()
        self._showing_raw = False
        self.raw_transcript_action.setChecked(False)
        self._set_audio_profile(AudioProfile.parse(session.audio_profile))
        self._display_transcript()
        self.speakers.set_session(session)
        self._dirty = False
        self.setWindowTitle("Tessitura — " + session.name)
        self._describe_session()
        self.save_text_btn.setEnabled(True)
        found = self._load_audio(session)
        note = "" if found else "  (audio non trovato su questo PC)"
        self.status_label.setText(f"Sessione caricata: {Path(path).name}{note}")
        # Aggiorna anche le azioni del menu (incluso il raggruppamento) dopo
        # aver sostituito la sessione corrente.
        self._set_busy(False)

    def _save_session(self) -> None:
        if self.session is None:
            QMessageBox.information(self, "Salva", "Niente da salvare.")
            return
        default = str(self.session_path or paths.sessions_dir() / f"{self.session.id}.json")
        path, _ = QFileDialog.getSaveFileName(self, "Salva sessione", default, "JSON (*.json)")
        if not path:
            return
        try:
            storage.save_session(self.session, path)
        except OSError as exc:
            QMessageBox.critical(self, "Salva sessione", str(exc))
            return
        learning = self._sync_project_learning()
        self.session_path = Path(path)
        self._dirty = False
        self.setWindowTitle("Tessitura — " + self.session.name)
        suffix = (
            f" · memoria progetto non aggiornata: {learning['error']}"
            if learning.get("error") else ""
        )
        self.status_label.setText(f"Salvato: {Path(path).name}{suffix}")

    def _export(self, kind: str) -> None:
        if self.session is None:
            QMessageBox.information(self, "Esporta", "Niente da esportare.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Esporta {kind.upper()}",
            str(paths.sessions_dir() / f"{self.session.id}.{kind}"),
            f"{kind.upper()} (*.{kind})",
        )
        if path:
            self._write_text(Path(path), kind)

    def _export_single_speaker(self, speaker_id: str) -> None:
        if self.session is None:
            return
        name = self.session.display_name(speaker_id)
        default = paths.sessions_dir() / f"{self.session.name or self.session.id}_{name}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Esporta trascrizione per {name}", str(default), "Testo (*.txt)"
        )
        if path:
            text = storage.export_txt_by_speaker(self.session, speaker_id)
            Path(path).write_text(text, encoding="utf-8")
            self.status_label.setText(f"Trascrizione per {name} salvata: {Path(path).name}")

    def _export_speakers_folder(self) -> None:
        if self.session is None:
            QMessageBox.information(self, "Esporta", "Nessuna sessione aperta.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Seleziona cartella di destinazione per le trascrizioni delle singole voci"
        )
        if folder:
            files = storage.export_all_speaker_transcripts(self.session, folder)
            self.status_label.setText(f"Esportati {len(files)} file di trascrizione per voce.")
            QMessageBox.information(
                self, "Esportazione completata",
                f"Esportate le trascrizioni per {len(files)} voci in:\n{folder}"
            )

    def _group_segments(self) -> None:
        """Accorpa i turni contigui della stessa voce, dopo conferma."""
        if self.session is None or not self.session.segments:
            QMessageBox.information(self, "Raggruppa testo", "Niente da raggruppare.")
            return
        available = self.session.adjacent_same_speaker_groups()
        if available <= 0:
            QMessageBox.information(
                self,
                "Raggruppa testo",
                "Non ci sono segmenti consecutivi attribuiti alla stessa voce.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Raggruppa testo",
            f"Sono disponibili {available} accorpamenti tra segmenti consecutivi.\n\n"
            "Vuoi unirli? I timestamp, le parole e la trascrizione grezza saranno conservati.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        merged = self.session.group_adjacent_speaker_segments()
        if not merged:
            self._set_busy(False)
            return
        self._showing_raw = False
        self.raw_transcript_action.setChecked(False)
        self._display_transcript()
        self.speakers.set_session(self.session)
        self._describe_session()
        self._mark_dirty()
        self._set_busy(False)
        self.status_label.setText(f"Raggruppati {merged} segmenti consecutivi della stessa voce.")

    def _check_updates(self) -> None:
        if self._updater is not None:
            return
        self.updates_btn.setEnabled(False)
        self.status_label.setText("Controllo aggiornamenti…")
        self._updater = UpdateCheckWorker(self)
        self._updater.progress.connect(self.status_label.setText)
        self._updater.done.connect(self._on_updates)
        self._updater.failed.connect(self._on_updates_failed)
        self._updater.finished.connect(self._updater.deleteLater)
        self._updater.finished.connect(lambda: setattr(self, "_updater", None))
        self._updater.start()

    def _on_updates(self, report) -> None:
        from .. import updates

        self.updates_btn.setEnabled(True)
        summary, detail = updates.format_report(report)
        self.status_label.setText(summary)
        box = QMessageBox(self)
        box.setWindowTitle("Aggiornamenti")
        box.setIcon(QMessageBox.Information)
        box.setText(summary)
        if detail:
            box.setDetailedText(detail)
        box.exec()

    def _on_updates_failed(self, message: str) -> None:
        self.updates_btn.setEnabled(True)
        self.status_label.setText("Controllo aggiornamenti non riuscito.")
        QMessageBox.warning(self, "Aggiornamenti", message)

    def _save_text(self) -> None:
        """Salva la trascrizione scegliendo il formato dall'estensione del file."""
        if self.session is None:
            QMessageBox.information(self, "Salva testo", "Niente da salvare.")
            return
        default = paths.sessions_dir() / f"{self.session.name or self.session.id}.txt"
        path, chosen = QFileDialog.getSaveFileName(
            self, "Salva testo trascritto", str(default),
            "Testo (*.txt);;Sottotitoli (*.srt)",
        )
        if not path:
            return
        out = Path(path)
        if not out.suffix:
            out = out.with_suffix(".srt" if "srt" in chosen.lower() else ".txt")
        self._write_text(out, "srt" if out.suffix.lower() == ".srt" else "txt")

    def _write_text(self, path: Path, kind: str) -> None:
        text = (
            storage.export_srt(self.session) if kind == "srt"
            else storage.export_txt(self.session)
        )
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Salva testo", str(exc))
            return
        self.status_label.setText(f"Salvato: {path.name}")

    def _open_model_settings(self) -> None:
        if self._busy or self._encoding or self._stopping or self.recorder.is_recording:
            return
        dialog = ModelSettingsDialog(
            self.hw,
            current_transcription=self.model_combo.currentText(),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self.model_combo.setCurrentText(dialog.selected_transcription_model())
        self._update_model_summary()
        self.status_label.setText(
            f"Impostazioni modelli salvate. Trascrizione: {self.model_combo.currentText()}"
        )

    def _update_model_summary(self) -> None:
        model = self.model_combo.currentText() if hasattr(self, "model_combo") else ""
        if hasattr(self, "model_summary_label"):
            self.model_summary_label.setText(model or "automatico")

    def _active_operation_state(self) -> ActiveOperationState:
        return ActiveOperationState(
            is_recording=bool(self.recorder.is_recording or self._stopping),
            is_combining=bool(self._encoding or (self._combiner is not None and self._combiner.isRunning())),
            is_processing=bool(
                self._busy
                or (self._worker is not None and self._worker.isRunning())
                or (self._batch_worker is not None and self._batch_worker.isRunning())
            ),
            is_exporting=False,
        )

    def _create_update_manager(self) -> UpdateManager:
        return UpdateManager(
            activity_provider=self._active_operation_state,
            quit_callback=self.close,
        )

    def _open_update_dialog(self) -> None:
        dialog = UpdateDialog(self._create_update_manager(), parent=self)
        dialog.exec()

    def _open_first_run_dialog(self) -> None:
        if self._active_operation_state().is_busy:
            self.status_label.setText("Attendi il termine dell'operazione in corso.")
            return
        dialog = FirstRunDialog(self.hw, parent=self)
        if dialog.exec() == QDialog.Accepted and dialog.open_model_manager_requested:
            self._open_model_settings()

    # --- cronologia delle modifiche -----------------------------------------

    def _undo_edit(self) -> None:
        if self.session is None or not self.session.undo():
            self.status_label.setText("Niente da annullare.")
            self._update_edit_actions()
            return
        self._after_history_change("Modifica annullata.")

    def _redo_edit(self) -> None:
        if self.session is None or not self.session.redo():
            self.status_label.setText("Niente da ripetere.")
            self._update_edit_actions()
            return
        self._after_history_change("Modifica ripetuta.")

    def _after_history_change(self, message: str) -> None:
        self._showing_raw = False
        if hasattr(self, "raw_transcript_action"):
            self.raw_transcript_action.setChecked(False)
        self._display_transcript()
        self.speakers.set_session(self.session)
        self._mark_dirty()
        self.status_label.setText(message)

    def _update_edit_actions(self) -> None:
        """Annulla/Ripeti seguono la cronologia della sessione aperta."""
        if not hasattr(self, "undo_action"):
            return
        session = self.session
        self.undo_action.setEnabled(bool(session and session.edit_history))
        self.redo_action.setEnabled(bool(session and session.redo_history))

    def closeEvent(self, event) -> None:
        if any(worker is not None and worker.isRunning()
               for worker in (self._worker, self._encoder, self._updater, self._stopper, getattr(self, "_region_worker", None), getattr(self, "_identity_worker", None))):
            self.status_label.setText("Attendi il termine dell'operazione prima di chiudere.")
            event.ignore()
            return
        if self.recorder.is_recording:
            self._stop_record()
            event.ignore()
            return
        if not self._confirm_discard():
            event.ignore()
            return
        self.player.clear()
        super().closeEvent(event)
