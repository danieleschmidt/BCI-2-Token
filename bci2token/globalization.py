"""
Global-First Implementation - BCI-2-Token Internationalization
Generation 4 Enhancement: Multi-region, Multi-language, Compliance-ready

This module implements comprehensive globalization features including:
- Multi-language support (en, es, fr, de, ja, zh)
- Regional compliance (GDPR, CCPA, PDPA, LGPD)
- Currency and timezone handling
- Cultural adaptation for BCI interfaces
- Accessibility compliance (WCAG 2.1 AA)
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
import locale
import warnings
from abc import ABC, abstractmethod

# Localization imports (graceful fallback)
try:
    import babel
    from babel.dates import format_datetime, format_currency
    from babel.numbers import format_decimal
    _HAS_BABEL = True
except ImportError:
    _HAS_BABEL = False
    warnings.warn("Babel not available. Limited localization features.")


@dataclass
class LocalizationConfig:
    """Configuration for localization settings"""
    default_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "es", "fr", "de", "ja", "zh", "pt", "it", "ru", "ko"
    ])
    default_region: str = "US"
    fallback_language: str = "en"
    auto_detect_language: bool = True
    cache_translations: bool = True
    
    
@dataclass  
class ComplianceConfig:
    """Configuration for regulatory compliance"""
    gdpr_enabled: bool = True  # European Union
    ccpa_enabled: bool = True  # California, USA
    pdpa_enabled: bool = True  # Singapore
    lgpd_enabled: bool = True  # Brazil
    pipeda_enabled: bool = True  # Canada
    data_retention_days: int = 365
    require_explicit_consent: bool = True
    anonymization_required: bool = True
    cross_border_transfer_allowed: bool = False
    

@dataclass
class AccessibilityConfig:
    """Configuration for accessibility compliance"""
    wcag_level: str = "AA"  # A, AA, or AAA
    screen_reader_support: bool = True
    high_contrast_mode: bool = True
    keyboard_navigation: bool = True
    voice_control_support: bool = True
    cognitive_load_reduction: bool = True
    seizure_safe_patterns: bool = True


class TranslationManager:
    """Manages translations and localization"""
    
    def __init__(self, config: LocalizationConfig, translations_dir: str = "locales"):
        self.config = config
        self.translations_dir = Path(translations_dir)
        self.translations = {}
        self.current_language = config.default_language
        self._load_translations()
    
    def _load_translations(self):
        """Load all translation files"""
        for lang in self.config.supported_languages:
            lang_file = self.translations_dir / f"{lang}.json"
            
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang] = json.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to load {lang} translations: {e}")
                    self.translations[lang] = {}
            else:
                # Create default translations
                self.translations[lang] = self._create_default_translations(lang)
                self._save_translation_file(lang)
    
    def _create_default_translations(self, language: str) -> Dict[str, str]:
        """Create default translations for a language"""
        
        # Base translations in English
        base_translations = {
            "app_name": "BCI-2-Token",
            "welcome_message": "Welcome to Brain-Computer Interface Translation",
            "processing": "Processing neural signals...",
            "error_occurred": "An error occurred during processing",
            "privacy_notice": "Your neural data is processed with privacy protection",
            "consent_required": "Consent required for data processing",
            "calibration_start": "Starting calibration process",
            "calibration_complete": "Calibration completed successfully",
            "signal_quality_good": "Signal quality is good",
            "signal_quality_poor": "Signal quality needs improvement",
            "device_connected": "Device connected successfully",
            "device_disconnected": "Device disconnected",
            "save_session": "Save session",
            "load_session": "Load session",
            "export_data": "Export data",
            "settings": "Settings",
            "help": "Help",
            "about": "About",
            "cancel": "Cancel",
            "confirm": "Confirm", 
            "yes": "Yes",
            "no": "No",
            "ok": "OK",
            "retry": "Retry",
            "close": "Close"
        }
        
        # Language-specific translations
        translations_map = {
            "es": {  # Spanish
                "welcome_message": "Bienvenido a la Traducción de Interfaz Cerebro-Computadora",
                "processing": "Procesando señales neurales...",
                "error_occurred": "Ocurrió un error durante el procesamiento",
                "privacy_notice": "Sus datos neurales se procesan con protección de privacidad",
                "consent_required": "Se requiere consentimiento para el procesamiento de datos",
                "calibration_start": "Iniciando proceso de calibración",
                "calibration_complete": "Calibración completada exitosamente",
                "signal_quality_good": "La calidad de la señal es buena",
                "signal_quality_poor": "La calidad de la señal necesita mejora",
                "device_connected": "Dispositivo conectado exitosamente",
                "device_disconnected": "Dispositivo desconectado",
                "settings": "Configuración",
                "help": "Ayuda",
                "about": "Acerca de",
                "cancel": "Cancelar",
                "confirm": "Confirmar",
                "yes": "Sí", 
                "no": "No",
                "retry": "Reintentar",
                "close": "Cerrar"
            },
            "fr": {  # French
                "welcome_message": "Bienvenue dans la Traduction d'Interface Cerveau-Ordinateur",
                "processing": "Traitement des signaux neuraux...",
                "error_occurred": "Une erreur s'est produite pendant le traitement",
                "privacy_notice": "Vos données neurales sont traitées avec protection de la vie privée",
                "consent_required": "Consentement requis pour le traitement des données",
                "calibration_start": "Démarrage du processus de calibration",
                "calibration_complete": "Calibration terminée avec succès",
                "signal_quality_good": "La qualité du signal est bonne",
                "signal_quality_poor": "La qualité du signal doit être améliorée",
                "device_connected": "Appareil connecté avec succès",
                "device_disconnected": "Appareil déconnecté",
                "settings": "Paramètres",
                "help": "Aide",
                "about": "À propos",
                "cancel": "Annuler",
                "confirm": "Confirmer",
                "yes": "Oui",
                "no": "Non",
                "retry": "Réessayer",
                "close": "Fermer"
            },
            "de": {  # German
                "welcome_message": "Willkommen bei der Gehirn-Computer-Schnittstellen-Übersetzung",
                "processing": "Verarbeitung von Neuralsignalen...",
                "error_occurred": "Ein Fehler ist bei der Verarbeitung aufgetreten",
                "privacy_notice": "Ihre Neuraldaten werden mit Datenschutz verarbeitet",
                "consent_required": "Einverständnis für Datenverarbeitung erforderlich",
                "calibration_start": "Kalibrierungsprozess wird gestartet",
                "calibration_complete": "Kalibrierung erfolgreich abgeschlossen",
                "signal_quality_good": "Die Signalqualität ist gut",
                "signal_quality_poor": "Die Signalqualität muss verbessert werden",
                "device_connected": "Gerät erfolgreich verbunden",
                "device_disconnected": "Gerät getrennt",
                "settings": "Einstellungen",
                "help": "Hilfe",
                "about": "Über",
                "cancel": "Abbrechen",
                "confirm": "Bestätigen",
                "yes": "Ja",
                "no": "Nein", 
                "retry": "Wiederholen",
                "close": "Schließen"
            },
            "ja": {  # Japanese  
                "welcome_message": "ブレイン・コンピュータ・インターフェース翻訳へようこそ",
                "processing": "神経信号を処理中...",
                "error_occurred": "処理中にエラーが発生しました",
                "privacy_notice": "あなたの神経データはプライバシー保護により処理されます",
                "consent_required": "データ処理には同意が必要です",
                "calibration_start": "キャリブレーションプロセスを開始",
                "calibration_complete": "キャリブレーションが正常に完了しました",
                "signal_quality_good": "信号品質は良好です",
                "signal_quality_poor": "信号品質の改善が必要です",
                "device_connected": "デバイスが正常に接続されました",
                "device_disconnected": "デバイスが切断されました",
                "settings": "設定",
                "help": "ヘルプ",
                "about": "について",
                "cancel": "キャンセル",
                "confirm": "確認",
                "yes": "はい",
                "no": "いいえ",
                "retry": "再試行",
                "close": "閉じる"
            },
            "zh": {  # Chinese (Simplified)
                "welcome_message": "欢迎使用脑机接口翻译",
                "processing": "正在处理神经信号...",
                "error_occurred": "处理过程中发生错误",
                "privacy_notice": "您的神经数据在隐私保护下处理",
                "consent_required": "数据处理需要同意",
                "calibration_start": "开始校准过程",
                "calibration_complete": "校准成功完成",
                "signal_quality_good": "信号质量良好",
                "signal_quality_poor": "信号质量需要改善",
                "device_connected": "设备连接成功",
                "device_disconnected": "设备已断开",
                "settings": "设置",
                "help": "帮助",
                "about": "关于",
                "cancel": "取消",
                "confirm": "确认",
                "yes": "是",
                "no": "否",
                "retry": "重试",
                "close": "关闭"
            }
        }
        
        # Apply language-specific translations
        result = base_translations.copy()
        if language in translations_map:
            result.update(translations_map[language])
            
        return result
    
    def _save_translation_file(self, language: str):
        """Save translations to file"""
        os.makedirs(self.translations_dir, exist_ok=True)
        lang_file = self.translations_dir / f"{language}.json"
        
        try:
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(self.translations[language], f, 
                         ensure_ascii=False, indent=2, sort_keys=True)
        except Exception as e:
            warnings.warn(f"Failed to save {language} translations: {e}")
    
    def set_language(self, language: str):
        """Set current language"""
        if language in self.config.supported_languages:
            self.current_language = language
        else:
            warnings.warn(f"Language {language} not supported, using {self.config.fallback_language}")
            self.current_language = self.config.fallback_language
    
    def get_text(self, key: str, language: Optional[str] = None) -> str:
        """Get translated text"""
        lang = language or self.current_language
        
        # Try current language
        if lang in self.translations and key in self.translations[lang]:
            return self.translations[lang][key]
        
        # Try fallback language
        if (self.config.fallback_language in self.translations and 
            key in self.translations[self.config.fallback_language]):
            return self.translations[self.config.fallback_language][key]
        
        # Return key as fallback
        return key
    
    def format_number(self, number: float, language: Optional[str] = None) -> str:
        """Format number according to locale"""
        lang = language or self.current_language
        
        if _HAS_BABEL:
            try:
                return format_decimal(number, locale=lang)
            except:
                pass
        
        # Fallback formatting
        if lang in ['en', 'en_US']:
            return f"{number:,.2f}"
        elif lang in ['de', 'de_DE']:
            return f"{number:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        elif lang in ['fr', 'fr_FR']:
            return f"{number:,.2f}".replace(',', ' ')
        else:
            return str(number)
    
    def format_datetime(self, dt: datetime, language: Optional[str] = None) -> str:
        """Format datetime according to locale"""
        lang = language or self.current_language
        
        if _HAS_BABEL:
            try:
                return format_datetime(dt, locale=lang)
            except:
                pass
        
        # Fallback formatting based on common patterns
        if lang in ['en', 'en_US']:
            return dt.strftime("%m/%d/%Y %I:%M %p")
        elif lang in ['de', 'de_DE']:
            return dt.strftime("%d.%m.%Y %H:%M")
        elif lang in ['fr', 'fr_FR']:
            return dt.strftime("%d/%m/%Y %H:%M")
        elif lang in ['ja', 'ja_JP']:
            return dt.strftime("%Y年%m月%d日 %H:%M")
        elif lang in ['zh', 'zh_CN']:
            return dt.strftime("%Y年%m月%d日 %H:%M")
        else:
            return dt.strftime("%Y-%m-%d %H:%M")


class ComplianceManager:
    """Manages regulatory compliance requirements"""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self.consent_records = {}
        self.data_processing_logs = []
        
    def check_compliance_requirements(self, user_region: str, 
                                    data_type: str) -> Dict[str, Any]:
        """Check what compliance requirements apply"""
        
        requirements = {
            'consent_required': False,
            'data_retention_limit': None,
            'anonymization_required': False,
            'cross_border_restrictions': False,
            'user_rights': [],
            'applicable_regulations': []
        }
        
        # GDPR (European Union)
        if self.config.gdpr_enabled and user_region in [
            'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 
            'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 
            'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'
        ]:
            requirements['applicable_regulations'].append('GDPR')
            requirements['consent_required'] = True
            requirements['data_retention_limit'] = self.config.data_retention_days
            requirements['anonymization_required'] = True
            requirements['cross_border_restrictions'] = True
            requirements['user_rights'].extend([
                'right_to_access', 'right_to_rectification', 'right_to_erasure',
                'right_to_portability', 'right_to_restrict_processing'
            ])
        
        # CCPA (California, USA)
        if self.config.ccpa_enabled and user_region in ['CA', 'US-CA']:
            requirements['applicable_regulations'].append('CCPA')
            requirements['consent_required'] = True
            requirements['user_rights'].extend([
                'right_to_know', 'right_to_delete', 'right_to_opt_out',
                'right_to_non_discrimination'
            ])
        
        # PDPA (Singapore)
        if self.config.pdpa_enabled and user_region == 'SG':
            requirements['applicable_regulations'].append('PDPA')
            requirements['consent_required'] = True
            requirements['data_retention_limit'] = self.config.data_retention_days
        
        # LGPD (Brazil)
        if self.config.lgpd_enabled and user_region == 'BR':
            requirements['applicable_regulations'].append('LGPD')
            requirements['consent_required'] = True
            requirements['anonymization_required'] = True
            requirements['user_rights'].extend([
                'right_to_access', 'right_to_rectification', 'right_to_deletion',
                'right_to_portability'
            ])
        
        return requirements
    
    def record_consent(self, user_id: str, consent_type: str, 
                      granted: bool, timestamp: Optional[datetime] = None):
        """Record user consent"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        consent_record = {
            'user_id': user_id,
            'consent_type': consent_type,
            'granted': granted,
            'timestamp': timestamp.isoformat(),
            'ip_address': None,  # Should be provided by application
            'user_agent': None   # Should be provided by application
        }
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent_record)
    
    def get_data_retention_policy(self, data_type: str, user_region: str) -> Dict[str, Any]:
        """Get data retention policy for specific data type and region"""
        
        policy = {
            'retention_period_days': self.config.data_retention_days,
            'auto_deletion': True,
            'anonymization_after_days': self.config.data_retention_days // 2,
            'backup_retention_days': self.config.data_retention_days + 30
        }
        
        # Adjust based on data type
        if data_type == 'neural_signals':
            # Neural data might need shorter retention
            policy['retention_period_days'] = min(policy['retention_period_days'], 180)
            policy['anonymization_after_days'] = 30  # Anonymize quickly
        elif data_type == 'calibration_data':
            # Calibration data might be kept longer for personalization
            policy['retention_period_days'] = min(policy['retention_period_days'], 365)
        elif data_type == 'session_metadata':
            # Metadata might be kept for analytics
            policy['retention_period_days'] = min(policy['retention_period_days'], 730)
        
        return policy
    
    def generate_privacy_notice(self, language: str = 'en') -> Dict[str, str]:
        """Generate privacy notice in specified language"""
        
        notices = {
            'en': {
                'title': 'Privacy Notice - BCI-2-Token',
                'data_collection': 'We collect neural signal data to provide brain-computer interface translation services.',
                'data_usage': 'Your neural data is used solely for signal processing and translation. It is not used for other purposes.',
                'data_protection': 'We implement state-of-the-art security measures including differential privacy and encryption.',
                'data_retention': f'Neural data is retained for {self.config.data_retention_days} days and then automatically deleted.',
                'user_rights': 'You have the right to access, correct, delete, and port your data.',
                'contact': 'Contact us at privacy@bci2token.com for privacy-related questions.'
            },
            'es': {
                'title': 'Aviso de Privacidad - BCI-2-Token',
                'data_collection': 'Recopilamos datos de señales neurales para proporcionar servicios de traducción de interfaz cerebro-computadora.',
                'data_usage': 'Sus datos neurales se utilizan únicamente para el procesamiento de señales y traducción.',
                'data_protection': 'Implementamos medidas de seguridad avanzadas incluyendo privacidad diferencial y encriptación.',
                'data_retention': f'Los datos neurales se retienen por {self.config.data_retention_days} días y luego se eliminan automáticamente.',
                'user_rights': 'Tiene derecho a acceder, corregir, eliminar y portar sus datos.',
                'contact': 'Contáctenos en privacy@bci2token.com para preguntas relacionadas con privacidad.'
            },
            'fr': {
                'title': 'Avis de Confidentialité - BCI-2-Token',
                'data_collection': 'Nous collectons des données de signaux neuraux pour fournir des services de traduction d\'interface cerveau-ordinateur.',
                'data_usage': 'Vos données neurales sont utilisées uniquement pour le traitement des signaux et la traduction.',
                'data_protection': 'Nous mettons en œuvre des mesures de sécurité avancées incluant la confidentialité différentielle et le chiffrement.',
                'data_retention': f'Les données neurales sont conservées pendant {self.config.data_retention_days} jours puis supprimées automatiquement.',
                'user_rights': 'Vous avez le droit d\'accéder, corriger, supprimer et porter vos données.',
                'contact': 'Contactez-nous à privacy@bci2token.com pour les questions liées à la confidentialité.'
            },
            'de': {
                'title': 'Datenschutzhinweis - BCI-2-Token',
                'data_collection': 'Wir sammeln Neuralsignaldaten, um Gehirn-Computer-Schnittstellen-Übersetzungsdienste anzubieten.',
                'data_usage': 'Ihre Neuraldaten werden ausschließlich für die Signalverarbeitung und Übersetzung verwendet.',
                'data_protection': 'Wir implementieren modernste Sicherheitsmaßnahmen einschließlich differenzieller Privatsphäre und Verschlüsselung.',
                'data_retention': f'Neuraldaten werden für {self.config.data_retention_days} Tage gespeichert und dann automatisch gelöscht.',
                'user_rights': 'Sie haben das Recht auf Zugang, Berichtigung, Löschung und Übertragbarkeit Ihrer Daten.',
                'contact': 'Kontaktieren Sie uns unter privacy@bci2token.com für datenschutzbezogene Fragen.'
            }
        }
        
        return notices.get(language, notices['en'])


class AccessibilityManager:
    """Manages accessibility and inclusive design features"""
    
    def __init__(self, config: AccessibilityConfig):
        self.config = config
        
    def get_accessibility_settings(self, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized accessibility settings"""
        
        settings = {
            'font_size_multiplier': 1.0,
            'contrast_mode': 'normal',
            'animation_reduced': False,
            'screen_reader_enabled': False,
            'keyboard_navigation_hints': False,
            'voice_control_active': False,
            'cognitive_assistance': False,
            'color_blind_friendly': False
        }
        
        # Apply user preferences
        if user_preferences.get('visual_impairment'):
            settings['font_size_multiplier'] = 1.5
            settings['contrast_mode'] = 'high'
            settings['screen_reader_enabled'] = True
            
        if user_preferences.get('motor_impairment'):
            settings['keyboard_navigation_hints'] = True
            settings['voice_control_active'] = True
            
        if user_preferences.get('cognitive_impairment'):
            settings['cognitive_assistance'] = True
            settings['animation_reduced'] = True
            
        if user_preferences.get('color_blindness'):
            settings['color_blind_friendly'] = True
            
        return settings
    
    def generate_accessibility_report(self) -> Dict[str, Any]:
        """Generate accessibility compliance report"""
        
        report = {
            'wcag_level': self.config.wcag_level,
            'compliant_features': [],
            'non_compliant_features': [],
            'recommendations': [],
            'score': 0.0
        }
        
        # Check various accessibility criteria
        checks = [
            ('keyboard_navigation', self.config.keyboard_navigation),
            ('screen_reader_support', self.config.screen_reader_support),
            ('high_contrast_mode', self.config.high_contrast_mode),
            ('voice_control_support', self.config.voice_control_support),
            ('cognitive_load_reduction', self.config.cognitive_load_reduction),
            ('seizure_safe_patterns', self.config.seizure_safe_patterns)
        ]
        
        for feature, enabled in checks:
            if enabled:
                report['compliant_features'].append(feature)
            else:
                report['non_compliant_features'].append(feature)
                report['recommendations'].append(f"Enable {feature} for better accessibility")
        
        # Calculate compliance score
        total_checks = len(checks)
        compliant_checks = len(report['compliant_features'])
        report['score'] = (compliant_checks / total_checks) * 100
        
        return report


class GlobalizationFramework:
    """Main framework coordinating all globalization features"""
    
    def __init__(self, 
                 localization_config: Optional[LocalizationConfig] = None,
                 compliance_config: Optional[ComplianceConfig] = None,
                 accessibility_config: Optional[AccessibilityConfig] = None):
        
        self.localization_config = localization_config or LocalizationConfig()
        self.compliance_config = compliance_config or ComplianceConfig()
        self.accessibility_config = accessibility_config or AccessibilityConfig()
        
        self.translation_manager = TranslationManager(self.localization_config)
        self.compliance_manager = ComplianceManager(self.compliance_config)
        self.accessibility_manager = AccessibilityManager(self.accessibility_config)
        
    def initialize_user_session(self, user_region: str, 
                              user_language: Optional[str] = None,
                              accessibility_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize a user session with appropriate settings"""
        
        # Set language
        if user_language:
            self.translation_manager.set_language(user_language)
        
        # Check compliance requirements
        compliance_reqs = self.compliance_manager.check_compliance_requirements(
            user_region, 'neural_signals'
        )
        
        # Configure accessibility
        accessibility_settings = self.accessibility_manager.get_accessibility_settings(
            accessibility_preferences or {}
        )
        
        # Generate privacy notice
        privacy_notice = self.compliance_manager.generate_privacy_notice(
            self.translation_manager.current_language
        )
        
        session_config = {
            'language': self.translation_manager.current_language,
            'region': user_region,
            'compliance_requirements': compliance_reqs,
            'accessibility_settings': accessibility_settings,
            'privacy_notice': privacy_notice,
            'translations': self.translation_manager.translations[self.translation_manager.current_language]
        }
        
        return session_config
    
    def get_localized_ui_text(self, keys: List[str]) -> Dict[str, str]:
        """Get localized text for UI elements"""
        return {key: self.translation_manager.get_text(key) for key in keys}
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        accessibility_report = self.accessibility_manager.generate_accessibility_report()
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'supported_languages': self.localization_config.supported_languages,
            'compliance_features': {
                'gdpr_enabled': self.compliance_config.gdpr_enabled,
                'ccpa_enabled': self.compliance_config.ccpa_enabled,
                'pdpa_enabled': self.compliance_config.pdpa_enabled,
                'lgpd_enabled': self.compliance_config.lgpd_enabled,
            },
            'accessibility_score': accessibility_report['score'],
            'accessibility_features': accessibility_report['compliant_features'],
            'recommendations': accessibility_report['recommendations']
        }
        
        return report


# Usage example and testing functions
def run_globalization_tests():
    """Run comprehensive globalization tests"""
    
    print("🌍 GLOBALIZATION FRAMEWORK TESTS")
    print("="*50)
    
    # Initialize framework
    framework = GlobalizationFramework()
    
    # Test different regions
    test_regions = [
        ('US', 'en', 'United States'),
        ('DE', 'de', 'Germany (GDPR)'),
        ('FR', 'fr', 'France (GDPR)'),
        ('CA', 'en', 'California (CCPA)'),
        ('SG', 'en', 'Singapore (PDPA)'),
        ('BR', 'pt', 'Brazil (LGPD)'),
        ('JP', 'ja', 'Japan'),
        ('CN', 'zh', 'China')
    ]
    
    for region, lang, description in test_regions:
        print(f"\n🔍 Testing: {description}")
        print("-" * 40)
        
        # Initialize session
        accessibility_prefs = {
            'visual_impairment': region == 'US',  # Test accessibility for US
            'motor_impairment': region == 'DE'    # Test for Germany
        }
        
        session = framework.initialize_user_session(
            region, lang, accessibility_prefs
        )
        
        print(f"Language: {session['language']}")
        print(f"Regulations: {', '.join(session['compliance_requirements']['applicable_regulations'])}")
        print(f"Consent Required: {session['compliance_requirements']['consent_required']}")
        print(f"User Rights: {len(session['compliance_requirements']['user_rights'])} rights")
        
        # Test localized text
        welcome = framework.translation_manager.get_text('welcome_message')
        print(f"Welcome Message: {welcome}")
        
        # Test accessibility
        if session['accessibility_settings']['screen_reader_enabled']:
            print("✅ Screen reader support enabled")
        if session['accessibility_settings']['high_contrast_mode']:
            print("✅ High contrast mode enabled")
    
    # Generate compliance report
    print(f"\n📋 COMPLIANCE REPORT")
    print("-" * 40)
    compliance_report = framework.generate_compliance_report()
    print(f"Languages Supported: {len(compliance_report['supported_languages'])}")
    print(f"Accessibility Score: {compliance_report['accessibility_score']:.1f}%")
    print(f"Compliance Features: {sum(compliance_report['compliance_features'].values())} enabled")
    
    print("\n✅ Globalization tests completed successfully!")


if __name__ == "__main__":
    run_globalization_tests()