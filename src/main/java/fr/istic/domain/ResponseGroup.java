package fr.istic.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import io.quarkus.hibernate.orm.panache.PanacheQuery;
import io.quarkus.panache.common.Parameters;
import jakarta.json.bind.annotation.JsonbTransient;
import io.quarkus.runtime.annotations.RegisterForReflection;

import jakarta.persistence.*;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * A Response Group.
 */
@Entity
@Table(name = "response_group")
@RegisterForReflection
public class ResponseGroup extends PanacheEntityBase implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Long id;

    @ManyToOne
    @JoinColumn(name = "question_id")
    @JsonbTransient
    public Question question;

    @Column(name="prediction_ids", columnDefinition = "text")
    private String predictionIdsJson;

    @Transient
    public Long[] predictionIds;

    @Column(name="average_embedding", columnDefinition = "longtext")
    private String averageEmbeddingJson;
    
    // Add a transient field that's used in your Java code
    @Transient
    public Double[] averageEmbedding;



    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ResponseGroup)) {
            return false;
        }
        return id != null && id.equals(((ResponseGroup) o).id);
    }

    @Override
    public int hashCode() {
        return 31;
    }

    @Override
    public String toString() {
        return "ResponseGroup{" +
            "id=" + id +
            ", question='" + question + "'" +
            ", predictionIds='" + predictionIds + "'" +
            ", averageEmbedding='" + averageEmbedding + "'" +
            "}";
    }
    
    // Convert JSON to array when loading from database
    @PostLoad
    void onLoad() {
        if (averageEmbeddingJson != null && !averageEmbeddingJson.isEmpty()) {
            try {
                ObjectMapper mapper = new ObjectMapper();
                averageEmbedding = mapper.readValue(averageEmbeddingJson, Double[].class);
            } catch (IOException e) {
                // Handle error or log it
                System.err.println("Error converting JSON to Double array: " + e.getMessage());
            }
        }
        if (predictionIdsJson != null && !predictionIdsJson.isEmpty()) {
            try {
                ObjectMapper mapper = new ObjectMapper();
                predictionIds = mapper.readValue(predictionIdsJson, Long[].class);
            } catch (IOException e) {
                System.err.println("Error converting JSON to Long array: " + e.getMessage());
            }
        }
    }
    
    // Convert array to JSON when saving to database
    @PrePersist
    @PreUpdate
    void onSave() {
        if (averageEmbedding != null) {
            try {
                ObjectMapper mapper = new ObjectMapper();
                averageEmbeddingJson = mapper.writeValueAsString(averageEmbedding);
            } catch (JsonProcessingException e) {
                // Handle error or log it
                System.err.println("Error converting Double array to JSON: " + e.getMessage());
            }
        }
        if (predictionIds != null) {
            try {
                ObjectMapper mapper = new ObjectMapper();
                predictionIdsJson = mapper.writeValueAsString(predictionIds);
            } catch (JsonProcessingException e) {
                System.err.println("Error converting Long array to JSON: " + e.getMessage());
            }
        }
    }
    
    // Add getter/setter for the averageEmbedding field
    public Double[] getAverageEmbedding() {
        return averageEmbedding;
    }
    
    public void setAverageEmbedding(Double[] averageEmbedding) {
        this.averageEmbedding = averageEmbedding;
    }

    public ResponseGroup update() {
        return update(this);
    }

    public ResponseGroup persistOrUpdate() {
        return persistOrUpdate(this);
    }

    public static ResponseGroup update(ResponseGroup responseGroup) {
        if (responseGroup == null) {
            throw new IllegalArgumentException("responseGroup can't be null");
        }
        var entity = ResponseGroup.<ResponseGroup>findById(responseGroup.id);
        if (entity != null) {
            entity.question = responseGroup.question;
            entity.predictionIds = responseGroup.predictionIds;
            entity.averageEmbedding = responseGroup.averageEmbedding;
        }
        return entity;
    }

    public static ResponseGroup persistOrUpdate(ResponseGroup responseGroup) {
        if (responseGroup == null) {
            throw new IllegalArgumentException("responseGroup can't be null");
        }
        if (responseGroup.id == null) {
            persist(responseGroup);
            return responseGroup;
        } else {
            return update(responseGroup);
        }
    }


    public static PanacheQuery<ResponseGroup> findByQuestionId(long qid) {
        return find("select rg from ResponseGroup rg where rg.question.id = ?1", qid);
    }
    
    public static long deleteByQIds(Set<Long> qids) {
        return delete("delete from ResponseGroup rg where rg.question.id in ?1", qids);
    }
    
    public static long deleteByQId(Long qid) {
        return delete("delete from ResponseGroup rg where rg.question.id = ?1", qid);
    }
    
    public static PanacheQuery<ResponseGroup> canAccess(long responseGroupId, String login) {
        return find("select rg from ResponseGroup rg join rg.question.exam.course.rgofs as u where rg.id = ?1 and u.login = ?2", responseGroupId, login);
    }
    
    public static PanacheQuery<ResponseGroup> findByQuestion(Long questionId) {
        return find("select rg from ResponseGroup rg where rg.question.id = ?1", questionId);
    }
    
public static PanacheQuery<ResponseGroup> findByPrediction(Long predictionId) {
    // Get all response groups
    List<ResponseGroup> allGroups = listAll();
    
    // Filter using Java
    List<ResponseGroup> filteredGroups = allGroups.stream()
        .filter(rg -> {
            if (rg.predictionIds == null) return false;
            for (Long id : rg.predictionIds) {
                if (id != null && id.equals(predictionId)) return true;
            }
            return false;
        })
        .collect(Collectors.toList());
    
    // Return as a PanacheQuery using find() with a filtered list id in clause
    if (filteredGroups.isEmpty()) {
        // Return an empty query if no matches
        return find("id = -1"); // This will return an empty result
    } else {
        // Get IDs of filtered groups
        List<Long> ids = filteredGroups.stream().map(rg -> rg.id).collect(Collectors.toList());
        return find("id in ?1", ids);
    }
}
}
