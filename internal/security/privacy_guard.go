// internal/security/privacy_guard.go
package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"io"
	
	"github.com/lumix-ai/vts/internal/core"
)

// PrivacyGuard - محافظ حریم خصوصی با رمزنگاری پیشرفته
type PrivacyGuard struct {
	encryptionEngine *AESGCMEngine
	anonymizer       *DataAnonymizer
	accessControl    *RBACController
	auditLogger      *AuditLogger
	complianceChecker *GDPRComplianceChecker
	
	encryptionKeys map[string][]byte
	dataPolicies   map[string]*DataPolicy
	userConsents   map[string]*ConsentRecord
}

// AESGCMEngine - موتور رمزنگاری AES-GCM
type AESGCMEngine struct {
	keyRotationInterval time.Duration
	currentKeyID        string
	keyStore            *SecureKeyStore
}

func (engine *AESGCMEngine) EncryptSensitiveData(data []byte, 
	dataType string) (*EncryptedData, error) {
	
	// انتخاب کلید مناسب بر اساس نوع داده
	keyID, key := engine.selectKey(dataType)
	
	// ایجاد nonce تصادفی
	nonce := make([]byte, 12)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}
	
	// ایجاد cipher با AES-GCM
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	
	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	
	// رمزنگاری داده
	ciphertext := aesgcm.Seal(nil, nonce, data, nil)
	
	return &EncryptedData{
		KeyID:      keyID,
		Ciphertext: ciphertext,
		Nonce:      nonce,
		DataType:   dataType,
		Timestamp:  time.Now(),
		IV:         nonce, // IV همان nonce در GCM
	}, nil
}

// DataAnonymizer - ناشناس‌ساز داده‌های حساس
type DataAnonymizer struct {
	techniques map[string]AnonymizationTechnique
	maskingRules []*MaskingRule
	pseudonymizer *PseudonymizationEngine
	differentialPrivacy *DifferentialPrivacyModule
}

func (da *DataAnonymizer) AnonymizeText(text string, 
	sensitivityLevel SensitivityLevel) (string, map[string]interface{}) {
	
	var anonymized string
	metadata := make(map[string]interface{})
	
	// شناسایی موجودیت‌های حساس
	entities := da.extractSensitiveEntities(text)
	
	// اعتماد تکنیک‌های ناشناس‌سازی بر اساس سطح حساسیت
	for _, entity := range entities {
		technique := da.selectTechnique(entity.Type, sensitivityLevel)
		
		switch technique {
		case "masking":
			anonymized = da.applyMasking(text, entity)
			metadata[entity.Type] = "masked"
			
		case "pseudonymization":
			pseudonym := da.pseudonymizer.GeneratePseudonym(entity.Value)
			anonymized = da.replaceEntity(text, entity, pseudonym)
			metadata[entity.Type] = "pseudonymized"
			metadata[entity.Type+"_hash"] = da.hash(entity.Value)
			
		case "generalization":
			generalized := da.generalizeEntity(entity)
			anonymized = da.replaceEntity(text, entity, generalized)
			metadata[entity.Type] = "generalized"
			
		case "differential_privacy":
			noisy := da.differentialPrivacy.AddNoise(entity.Value)
			anonymized = da.replaceEntity(text, entity, noisy)
			metadata[entity.Type] = "differentially_private"
			
		case "redaction":
			anonymized = da.redactEntity(text, entity)
			metadata[entity.Type] = "redacted"
		}
		
		text = anonymized
	}
	
	// افزودن نویز برای جلوگیری از شناسایی
	if sensitivityLevel == HighSensitivity {
		text = da.addNoise(text)
	}
	
	return text, metadata
}

// DifferentialPrivacyModule - حریم خصوصی تفاضلی
type DifferentialPrivacyModule struct {
	epsilon     float64
	delta       float64
	noiseType   string // "laplace", "gaussian"
	sensitivity float64
	randomizer  *SecureRandomizer
}

func (dp *DifferentialPrivacyModule) AddNoise(value float64) float64 {
	var noise float64
	
	switch dp.noiseType {
	case "laplace":
		noise = dp.laplaceNoise()
	case "gaussian":
		noise = dp.gaussianNoise()
	default:
		noise = dp.laplaceNoise()
	}
	
	return value + noise
}

func (dp *DifferentialPrivacyModule) laplaceNoise() float64 {
	// تولید نویز لاپلاس
	scale := dp.sensitivity / dp.epsilon
	u := dp.randomizer.Uniform(-0.5, 0.5)
	
	return -scale * math.Copysign(1.0, u) * math.Log(1-2*math.Abs(u))
}

// SecureDataStorage - ذخیره‌سازی امن داده‌ها
type SecureDataStorage struct {
	encryptedFS   *EncryptedFileSystem
	secureDB      *EncryptedDatabase
	keyManagement *KeyManagementSystem
	backupManager *EncryptedBackupManager
	integrityChecker *DataIntegrityChecker
}

func (sds *SecureDataStorage) StoreUserData(userID string, 
	data *UserData, consent *ConsentRecord) error {
	
	// بررسی رضایت کاربر
	if !consent.IsValidFor(data.DataType) {
		return fmt.Errorf("user consent required for %s", data.DataType)
	}
	
	// رمزنگاری داده
	encryptedData, err := sds.encryptData(data.RawData, userID)
	if err != nil {
		return err
	}
	
	// ذخیره در فایل‌سیستم رمزنگاری شده
	filePath := sds.encryptedFS.Store(encryptedData, userID, data.DataType)
	
	// ذخیره فراداده در پایگاه داده امن
	metadata := &DataMetadata{
		UserID:     userID,
		DataType:   data.DataType,
		FilePath:   filePath,
		EncryptionKeyID: encryptedData.KeyID,
		ConsentID:  consent.ID,
		RetentionPeriod: consent.RetentionPeriod,
		AccessPolicy: data.AccessPolicy,
	}
	
	if err := sds.secureDB.StoreMetadata(metadata); err != nil {
		// حذف فایل اگر ذخیره فراداده ناموفق بود
		sds.encryptedFS.Delete(filePath)
		return err
	}
	
	// ایجاد بک‌آپ رمزنگاری شده
	go sds.backupManager.CreateBackup(filePath, userID)
	
	// تأیید یکپارچگی داده
	go sds.integrityChecker.VerifyIntegrity(filePath, encryptedData.Checksum)
	
	return nil
}