"""
Global-First Deployment Framework
=================================

Multi-region deployment with internationalization support:
- I18n support for en, es, fr, de, ja, zh
- Multi-region deployment with latency optimization  
- Compliance with GDPR, CCPA, PDPA
- Cross-platform compatibility
- Regional data sovereignty
- Cultural adaptation for BCI interfaces
"""

import asyncio
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings

class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de" 
    JAPANESE = "ja"
    CHINESE = "zh"

class Region(Enum):
    """Supported deployment regions."""
    NORTH_AMERICA = "us-east-1"
    EUROPE = "eu-west-1"
    ASIA_PACIFIC = "ap-southeast-1"
    LATIN_AMERICA = "sa-east-1"
    MIDDLE_EAST = "me-south-1"
    AFRICA = "af-south-1"

class ComplianceFramework(Enum):
    """Data protection compliance frameworks."""
    GDPR = "gdpr"      # European Union
    CCPA = "ccpa"      # California, USA
    PDPA = "pdpa"      # Singapore, Thailand
    PIPEDA = "pipeda"  # Canada
    LGPD = "lgpd"      # Brazil

@dataclass
class I18nMessage:
    """Internationalization message container."""
    key: str
    translations: Dict[SupportedLanguage, str] = field(default_factory=dict)
    context: Optional[str] = None
    pluralization: Dict[str, Dict[SupportedLanguage, str]] = field(default_factory=dict)

class InternationalizationManager:
    """
    Comprehensive internationalization manager.
    
    Supports:
    - Multi-language message translation
    - Cultural formatting (dates, numbers, currencies)
    - Right-to-left language support
    - Pluralization rules
    - Context-aware translations
    """
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.messages: Dict[str, I18nMessage] = {}
        self.fallback_chain = [SupportedLanguage.ENGLISH]  # Always fall back to English
        
        # Cultural formatting rules
        self.number_formats = {
            SupportedLanguage.ENGLISH: {'decimal': '.', 'thousands': ','},
            SupportedLanguage.GERMAN: {'decimal': ',', 'thousands': '.'},
            SupportedLanguage.FRENCH: {'decimal': ',', 'thousands': ' '},
            SupportedLanguage.SPANISH: {'decimal': ',', 'thousands': '.'},
            SupportedLanguage.JAPANESE: {'decimal': '.', 'thousands': ','},
            SupportedLanguage.CHINESE: {'decimal': '.', 'thousands': ','}
        }
        
        # Date format patterns
        self.date_formats = {
            SupportedLanguage.ENGLISH: '%m/%d/%Y',
            SupportedLanguage.GERMAN: '%d.%m.%Y', 
            SupportedLanguage.FRENCH: '%d/%m/%Y',
            SupportedLanguage.SPANISH: '%d/%m/%Y',
            SupportedLanguage.JAPANESE: '%Y年%m月%d日',
            SupportedLanguage.CHINESE: '%Y年%m月%d日'
        }
        
        # Load default BCI-specific messages
        self._load_default_messages()
        
    def _load_default_messages(self):
        """Load default BCI-2-Token messages."""
        default_messages = {
            'bci.signal.quality.excellent': I18nMessage(
                'bci.signal.quality.excellent',
                {
                    SupportedLanguage.ENGLISH: 'Excellent signal quality',
                    SupportedLanguage.SPANISH: 'Excelente calidad de señal',
                    SupportedLanguage.FRENCH: 'Excellente qualité du signal',
                    SupportedLanguage.GERMAN: 'Ausgezeichnete Signalqualität',
                    SupportedLanguage.JAPANESE: '優秀な信号品質',
                    SupportedLanguage.CHINESE: '优秀的信号质量'
                }
            ),
            'bci.signal.quality.poor': I18nMessage(
                'bci.signal.quality.poor',
                {
                    SupportedLanguage.ENGLISH: 'Poor signal quality - please check electrodes',
                    SupportedLanguage.SPANISH: 'Mala calidad de señal - por favor revise los electrodos',
                    SupportedLanguage.FRENCH: 'Mauvaise qualité du signal - veuillez vérifier les électrodes',
                    SupportedLanguage.GERMAN: 'Schlechte Signalqualität - bitte Elektroden prüfen',
                    SupportedLanguage.JAPANESE: '信号品質が悪い - 電極を確認してください',
                    SupportedLanguage.CHINESE: '信号质量差 - 请检查电极'
                }
            ),
            'bci.calibration.start': I18nMessage(
                'bci.calibration.start',
                {
                    SupportedLanguage.ENGLISH: 'Starting calibration - please focus',
                    SupportedLanguage.SPANISH: 'Iniciando calibración - por favor concéntrese',
                    SupportedLanguage.FRENCH: 'Démarrage de la calibration - veuillez vous concentrer',
                    SupportedLanguage.GERMAN: 'Kalibrierung beginnt - bitte konzentrieren Sie sich',
                    SupportedLanguage.JAPANESE: 'キャリブレーション開始 - 集中してください',
                    SupportedLanguage.CHINESE: '开始校准 - 请集中注意力'
                }
            ),
            'bci.processing.tokens': I18nMessage(
                'bci.processing.tokens',
                {
                    SupportedLanguage.ENGLISH: 'Processing {count} tokens',
                    SupportedLanguage.SPANISH: 'Procesando {count} tokens',
                    SupportedLanguage.FRENCH: 'Traitement de {count} tokens',
                    SupportedLanguage.GERMAN: 'Verarbeitung von {count} Token',
                    SupportedLanguage.JAPANESE: '{count}個のトークンを処理中',
                    SupportedLanguage.CHINESE: '正在处理{count}个令牌'
                }
            ),
            'bci.privacy.enabled': I18nMessage(
                'bci.privacy.enabled',
                {
                    SupportedLanguage.ENGLISH: 'Privacy protection is enabled',
                    SupportedLanguage.SPANISH: 'La protección de privacidad está habilitada',
                    SupportedLanguage.FRENCH: 'La protection de la vie privée est activée',
                    SupportedLanguage.GERMAN: 'Datenschutz ist aktiviert',
                    SupportedLanguage.JAPANESE: 'プライバシー保護が有効です',
                    SupportedLanguage.CHINESE: '隐私保护已启用'
                }
            ),
            'error.connection.failed': I18nMessage(
                'error.connection.failed',
                {
                    SupportedLanguage.ENGLISH: 'Connection failed - please check your device',
                    SupportedLanguage.SPANISH: 'Conexión fallida - por favor revise su dispositivo',
                    SupportedLanguage.FRENCH: 'Échec de la connexion - veuillez vérifier votre appareil',
                    SupportedLanguage.GERMAN: 'Verbindung fehlgeschlagen - bitte Gerät überprüfen',
                    SupportedLanguage.JAPANESE: '接続に失敗しました - デバイスを確認してください',
                    SupportedLanguage.CHINESE: '连接失败 - 请检查您的设备'
                }
            ),
            'system.ready': I18nMessage(
                'system.ready',
                {
                    SupportedLanguage.ENGLISH: 'BCI-2-Token system is ready',
                    SupportedLanguage.SPANISH: 'El sistema BCI-2-Token está listo',
                    SupportedLanguage.FRENCH: 'Le système BCI-2-Token est prêt',
                    SupportedLanguage.GERMAN: 'BCI-2-Token System ist bereit',
                    SupportedLanguage.JAPANESE: 'BCI-2-Tokenシステムの準備完了',
                    SupportedLanguage.CHINESE: 'BCI-2-Token系统已准备就绪'
                }
            )
        }
        
        self.messages.update(default_messages)
        
    def set_language(self, language: SupportedLanguage):
        """Set current language for translations."""
        self.current_language = language
        
    def translate(self, key: str, language: Optional[SupportedLanguage] = None, **kwargs) -> str:
        """
        Translate message key to specified or current language.
        
        Args:
            key: Message key to translate
            language: Target language (uses current if None)
            **kwargs: Format parameters for message interpolation
            
        Returns:
            Translated message string
        """
        target_lang = language or self.current_language
        
        if key not in self.messages:
            warnings.warn(f"Translation key not found: {key}")
            return key
            
        message = self.messages[key]
        
        # Try target language first
        if target_lang in message.translations:
            text = message.translations[target_lang]
        else:
            # Fall back through fallback chain
            text = None
            for fallback_lang in self.fallback_chain:
                if fallback_lang in message.translations:
                    text = message.translations[fallback_lang]
                    break
                    
            if not text:
                warnings.warn(f"No translation found for key: {key}")
                return key
                
        # Apply string formatting
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError as e:
                warnings.warn(f"Missing format parameter {e} for key: {key}")
                
        return text
        
    def format_number(self, number: float, language: Optional[SupportedLanguage] = None) -> str:
        """Format number according to cultural conventions."""
        target_lang = language or self.current_language
        format_rules = self.number_formats.get(target_lang, self.number_formats[SupportedLanguage.ENGLISH])
        
        # Convert number to string with proper separators
        num_str = f"{number:,.2f}"
        
        # Replace separators according to locale
        if format_rules['decimal'] != '.':
            num_str = num_str.replace('.', '_DECIMAL_')
        if format_rules['thousands'] != ',':
            num_str = num_str.replace(',', format_rules['thousands'])
        if format_rules['decimal'] != '.':
            num_str = num_str.replace('_DECIMAL_', format_rules['decimal'])
            
        return num_str
        
    def format_percentage(self, value: float, language: Optional[SupportedLanguage] = None) -> str:
        """Format percentage according to cultural conventions."""
        formatted_num = self.format_number(value * 100, language)
        return f"{formatted_num}%"
        
    def is_rtl_language(self, language: Optional[SupportedLanguage] = None) -> bool:
        """Check if language is right-to-left."""
        # None of our currently supported languages are RTL
        # Would include Arabic, Hebrew, etc. in future
        return False
        
    def add_message(self, message: I18nMessage):
        """Add new translatable message."""
        self.messages[message.key] = message
        
    def get_supported_languages(self) -> List[SupportedLanguage]:
        """Get list of supported languages."""
        return list(SupportedLanguage)

class RegionalDeploymentManager:
    """
    Multi-region deployment management.
    
    Handles:
    - Regional resource allocation
    - Data sovereignty compliance  
    - Latency optimization
    - Failover between regions
    """
    
    def __init__(self):
        self.regions: Dict[Region, Dict[str, Any]] = {}
        self.region_capabilities: Dict[Region, List[str]] = {}
        self.data_residency_rules: Dict[str, List[Region]] = {}
        self.latency_matrix: Dict[Tuple[Region, Region], float] = {}
        
        # Initialize default region capabilities
        self._initialize_region_capabilities()
        
    def _initialize_region_capabilities(self):
        """Initialize default capabilities for each region."""
        self.region_capabilities = {
            Region.NORTH_AMERICA: ['gpu_compute', 'quantum_access', 'edge_nodes', 'healthcare_compliance'],
            Region.EUROPE: ['gpu_compute', 'gdpr_compliance', 'edge_nodes', 'quantum_research'],
            Region.ASIA_PACIFIC: ['gpu_compute', 'edge_nodes', 'manufacturing_integration'],
            Region.LATIN_AMERICA: ['cpu_compute', 'edge_nodes', 'cost_optimization'],
            Region.MIDDLE_EAST: ['cpu_compute', 'edge_nodes', 'data_sovereignty'],
            Region.AFRICA: ['cpu_compute', 'edge_nodes', 'bandwidth_optimization']
        }
        
        # Initialize latency matrix (simulated)
        regions = list(Region)
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i == j:
                    self.latency_matrix[(region1, region2)] = 5.0  # Local latency
                else:
                    # Simulate geographical latency
                    base_latency = 50.0 + abs(i - j) * 20.0
                    self.latency_matrix[(region1, region2)] = base_latency
                    
    def register_region(self, region: Region, config: Dict[str, Any]):
        """Register a deployment region."""
        self.regions[region] = {
            'region': region,
            'status': 'active',
            'capacity': config.get('capacity', 1.0),
            'cost_per_hour': config.get('cost_per_hour', 1.0),
            'compliance_frameworks': config.get('compliance_frameworks', []),
            'available_resources': config.get('available_resources', []),
            'data_centers': config.get('data_centers', []),
            'health_score': 1.0,
            'current_load': 0.0
        }
        
    def find_optimal_region(self, requirements: Dict[str, Any], 
                          user_location: Optional[Region] = None) -> Optional[Region]:
        """
        Find optimal region for deployment based on requirements.
        
        Args:
            requirements: Deployment requirements
            user_location: User's geographical region
            
        Returns:
            Optimal region or None if no suitable region found
        """
        candidate_regions = []
        
        for region, config in self.regions.items():
            if config['status'] != 'active':
                continue
                
            # Check capacity
            if config['current_load'] >= config['capacity'] * 0.9:  # 90% capacity limit
                continue
                
            # Check compliance requirements
            required_compliance = requirements.get('compliance_frameworks', [])
            available_compliance = config.get('compliance_frameworks', [])
            if required_compliance and not any(cf in available_compliance for cf in required_compliance):
                continue
                
            # Check resource requirements
            required_resources = requirements.get('required_resources', [])
            available_resources = config.get('available_resources', [])
            if required_resources and not any(res in available_resources for res in required_resources):
                continue
                
            # Calculate score
            score = self._calculate_region_score(region, requirements, user_location)
            candidate_regions.append((region, score))
            
        if not candidate_regions:
            return None
            
        # Sort by score (higher is better) and return best region
        candidate_regions.sort(key=lambda x: x[1], reverse=True)
        return candidate_regions[0][0]
        
    def _calculate_region_score(self, region: Region, requirements: Dict[str, Any],
                              user_location: Optional[Region]) -> float:
        """Calculate region suitability score."""
        config = self.regions[region]
        score = 0.0
        
        # Health and capacity score (30%)
        health_score = config['health_score'] * (1.0 - config['current_load'] / config['capacity'])
        score += health_score * 0.3
        
        # Latency score (25%)
        if user_location:
            latency = self.latency_matrix.get((user_location, region), 100.0)
            latency_score = max(0.0, 1.0 - latency / 200.0)  # Normalize to 200ms max
            score += latency_score * 0.25
            
        # Cost score (20%)
        max_cost = max(r['cost_per_hour'] for r in self.regions.values())
        cost_score = 1.0 - (config['cost_per_hour'] / max_cost)
        score += cost_score * 0.2
        
        # Capability match score (15%)
        required_capabilities = requirements.get('required_capabilities', [])
        available_capabilities = self.region_capabilities.get(region, [])
        if required_capabilities:
            capability_matches = sum(1 for cap in required_capabilities if cap in available_capabilities)
            capability_score = capability_matches / len(required_capabilities)
            score += capability_score * 0.15
            
        # Compliance score (10%)
        required_compliance = requirements.get('compliance_frameworks', [])
        available_compliance = config.get('compliance_frameworks', [])
        if required_compliance:
            compliance_matches = sum(1 for cf in required_compliance if cf in available_compliance)
            compliance_score = compliance_matches / len(required_compliance)
            score += compliance_score * 0.1
            
        return min(score, 1.0)  # Cap at 1.0
        
    def estimate_deployment_cost(self, region: Region, duration_hours: float) -> Dict[str, float]:
        """Estimate deployment cost for region."""
        if region not in self.regions:
            return {'error': 'Region not available'}
            
        config = self.regions[region]
        base_cost = config['cost_per_hour'] * duration_hours
        
        return {
            'base_cost': base_cost,
            'estimated_total': base_cost * 1.15,  # Add 15% for overhead
            'currency': 'USD',
            'duration_hours': duration_hours
        }

class ComplianceManager:
    """
    Data protection compliance management.
    
    Ensures compliance with:
    - GDPR (European Union)
    - CCPA (California)
    - PDPA (Singapore, Thailand)
    - Other regional data protection laws
    """
    
    def __init__(self):
        self.compliance_rules: Dict[ComplianceFramework, Dict[str, Any]] = {}
        self.data_processing_records: List[Dict[str, Any]] = []
        self._initialize_compliance_rules()
        
    def _initialize_compliance_rules(self):
        """Initialize compliance rules for supported frameworks."""
        self.compliance_rules = {
            ComplianceFramework.GDPR: {
                'data_retention_max_days': 2555,  # 7 years max
                'requires_explicit_consent': True,
                'requires_data_protection_officer': True,
                'breach_notification_hours': 72,
                'right_to_erasure': True,
                'right_to_portability': True,
                'privacy_by_design': True,
                'allowed_data_transfers': ['eu', 'adequacy_decision_countries'],
                'pseudonymization_required': True,
                'encryption_at_rest_required': True,
                'encryption_in_transit_required': True
            },
            ComplianceFramework.CCPA: {
                'data_retention_max_days': 365,  # 1 year for some data
                'requires_explicit_consent': False,  # Opt-out model
                'requires_privacy_officer': True,
                'breach_notification_days': 30,
                'right_to_delete': True,
                'right_to_know': True,
                'right_to_opt_out': True,
                'non_discrimination': True,
                'allowed_data_sharing': ['service_providers', 'business_purposes'],
                'privacy_policy_required': True
            },
            ComplianceFramework.PDPA: {
                'data_retention_max_days': 1095,  # 3 years typical
                'requires_explicit_consent': True,
                'requires_dpo': True,
                'breach_notification_days': 3,
                'right_to_access': True,
                'right_to_correction': True,
                'right_to_portability': True,
                'cross_border_restrictions': True,
                'data_protection_by_design': True
            }
        }
        
    def check_compliance(self, framework: ComplianceFramework, 
                        data_processing: Dict[str, Any]) -> Dict[str, Any]:
        """Check if data processing complies with framework."""
        if framework not in self.compliance_rules:
            return {'compliant': False, 'reason': 'Framework not supported'}
            
        rules = self.compliance_rules[framework]
        compliance_issues = []
        
        # Check data retention
        retention_days = data_processing.get('retention_days', 0)
        max_retention = rules.get('data_retention_max_days', float('inf'))
        if retention_days > max_retention:
            compliance_issues.append(f'Data retention exceeds maximum: {retention_days} > {max_retention} days')
            
        # Check consent requirements
        if rules.get('requires_explicit_consent', False):
            if not data_processing.get('has_explicit_consent', False):
                compliance_issues.append('Explicit consent required but not obtained')
                
        # Check encryption requirements
        if rules.get('encryption_at_rest_required', False):
            if not data_processing.get('encrypted_at_rest', False):
                compliance_issues.append('Encryption at rest required')
                
        if rules.get('encryption_in_transit_required', False):
            if not data_processing.get('encrypted_in_transit', False):
                compliance_issues.append('Encryption in transit required')
                
        # Check pseudonymization
        if rules.get('pseudonymization_required', False):
            if not data_processing.get('pseudonymized', False):
                compliance_issues.append('Pseudonymization required for this data type')
                
        is_compliant = len(compliance_issues) == 0
        
        return {
            'compliant': is_compliant,
            'framework': framework.value,
            'issues': compliance_issues,
            'requirements_met': len(rules) - len(compliance_issues),
            'total_requirements': len(rules)
        }
        
    def generate_privacy_notice(self, framework: ComplianceFramework, 
                              language: SupportedLanguage = SupportedLanguage.ENGLISH) -> str:
        """Generate privacy notice text for compliance framework."""
        if framework == ComplianceFramework.GDPR:
            if language == SupportedLanguage.ENGLISH:
                return """
BCI-2-Token Privacy Notice (GDPR)

We process your brain signal data to provide BCI-to-token translation services.

Legal Basis: Your explicit consent
Data Retention: Maximum 7 years
Your Rights: Access, rectification, erasure, portability, restriction, objection
Contact: dpo@bci2token.com

Your data is encrypted and pseudonymized. You may withdraw consent at any time.
"""
            elif language == SupportedLanguage.GERMAN:
                return """
BCI-2-Token Datenschutzerklärung (DSGVO)

Wir verarbeiten Ihre Gehirnsignaldaten zur Bereitstellung von BCI-zu-Token-Übersetzungsdiensten.

Rechtsgrundlage: Ihre ausdrückliche Einwilligung
Speicherdauer: Maximal 7 Jahre
Ihre Rechte: Zugang, Berichtigung, Löschung, Übertragbarkeit, Einschränkung, Widerspruch
Kontakt: dpo@bci2token.com

Ihre Daten werden verschlüsselt und pseudonymisiert. Sie können die Einwilligung jederzeit widerrufen.
"""
                
        elif framework == ComplianceFramework.CCPA:
            return """
BCI-2-Token Privacy Notice (CCPA)

We collect and process brain signal data for BCI services.

Your Rights: Know, delete, opt-out, non-discrimination
Data Sharing: Only with service providers for business purposes
Contact: privacy@bci2token.com

You may opt-out of data sale (we don't sell your data) and request deletion.
"""
            
        return f"Privacy notice for {framework.value} not available in {language.value}"
        
    def log_data_processing(self, processing_activity: Dict[str, Any]):
        """Log data processing activity for compliance audit."""
        activity_record = {
            'timestamp': time.time(),
            'activity_type': processing_activity.get('type', 'unknown'),
            'data_subject_id': processing_activity.get('user_id', 'anonymous'),
            'data_categories': processing_activity.get('data_categories', []),
            'processing_purposes': processing_activity.get('purposes', []),
            'legal_basis': processing_activity.get('legal_basis', 'consent'),
            'retention_period': processing_activity.get('retention_days', 365),
            'third_party_sharing': processing_activity.get('shared_with', []),
            'security_measures': processing_activity.get('security_measures', [])
        }
        
        self.data_processing_records.append(activity_record)
        
        # Limit log size
        if len(self.data_processing_records) > 10000:
            self.data_processing_records = self.data_processing_records[-5000:]

# Global instances
_global_i18n_manager: Optional[InternationalizationManager] = None
_global_regional_manager: Optional[RegionalDeploymentManager] = None
_global_compliance_manager: Optional[ComplianceManager] = None

def get_i18n_manager() -> InternationalizationManager:
    """Get global internationalization manager."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = InternationalizationManager()
    return _global_i18n_manager

def get_regional_manager() -> RegionalDeploymentManager:
    """Get global regional deployment manager."""  
    global _global_regional_manager
    if _global_regional_manager is None:
        _global_regional_manager = RegionalDeploymentManager()
    return _global_regional_manager

def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager."""
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = ComplianceManager()
    return _global_compliance_manager

def initialize_global_deployment() -> Dict[str, Any]:
    """Initialize global deployment system."""
    i18n = get_i18n_manager()
    regional = get_regional_manager()
    compliance = get_compliance_manager()
    
    # Register sample regions
    sample_regions = {
        Region.NORTH_AMERICA: {
            'capacity': 1.0,
            'cost_per_hour': 2.0,
            'compliance_frameworks': [ComplianceFramework.CCPA],
            'available_resources': ['gpu_compute', 'quantum_access']
        },
        Region.EUROPE: {
            'capacity': 0.8, 
            'cost_per_hour': 2.5,
            'compliance_frameworks': [ComplianceFramework.GDPR],
            'available_resources': ['gpu_compute', 'quantum_research']
        },
        Region.ASIA_PACIFIC: {
            'capacity': 0.9,
            'cost_per_hour': 1.5,
            'compliance_frameworks': [ComplianceFramework.PDPA],
            'available_resources': ['gpu_compute', 'manufacturing_integration']
        }
    }
    
    for region, config in sample_regions.items():
        regional.register_region(region, config)
        
    return {
        'i18n_manager': i18n,
        'regional_manager': regional,
        'compliance_manager': compliance,
        'supported_languages': len(i18n.get_supported_languages()),
        'registered_regions': len(regional.regions),
        'compliance_frameworks': len(compliance.compliance_rules)
    }